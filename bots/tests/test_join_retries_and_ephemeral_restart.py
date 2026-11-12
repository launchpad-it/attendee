from unittest.mock import MagicMock, patch

import docker
from django.test import SimpleTestCase
from django.utils import timezone

from bots.bot_adapter import BotAdapter
from bots.bot_controller import BotController
from bots.google_meet_bot_adapter.google_meet_bot_adapter import GoogleMeetBotAdapter
from bots.google_meet_bot_adapter.google_meet_ui_methods import UiGoogleBlockingUsException, UiMocapSequenceNotAvailableException
from bots.models import BotEventTypes
from bots.tasks import restart_bot_in_ephemeral_container_task as restart_task
from bots.tasks.restart_bot_in_ephemeral_container_task import restart_bot_in_ephemeral_container
from bots.web_bot_adapter import web_bot_adapter

BLOCKED = {"message": BotAdapter.Messages.BLOCKED_BY_PLATFORM_REPEATEDLY}


def google_block():
    return UiGoogleBlockingUsException("You can't join this video call", "check_if_blocked")


def mocap_dead_end():
    return UiMocapSequenceNotAvailableException("No mocap sequence lands inside clickable rect", "join_button")


def make_adapter(join_outcomes):
    adapter = GoogleMeetBotAdapter.__new__(GoogleMeetBotAdapter)
    adapter.meeting_url = "https://meet.google.com/abc-defg-hij"
    adapter.left_meeting = False
    adapter.cleaned_up = False
    adapter.automatic_leave_configuration = MagicMock()
    adapter.init_driver = MagicMock()
    adapter.attempt_to_join_meeting = MagicMock(side_effect=join_outcomes)
    adapter.send_message_callback = MagicMock()
    adapter.send_debug_screenshot_message = MagicMock()
    adapter.after_bot_joined_meeting = MagicMock()
    adapter.subclass_specific_after_bot_joined_meeting = MagicMock()
    return adapter


def blocked_messages_sent(adapter):
    return [c for c in adapter.send_message_callback.call_args_list if c.args == (BLOCKED,)]


@patch.object(web_bot_adapter, "sleep", MagicMock())
class TestJoinRetryCounting(SimpleTestCase):
    def test_joins_after_mouse_dead_ends_mixed_with_three_google_blocks(self):
        # The exact sequence that made production bots give up on 2026-11-09 and 2026-11-11.
        adapter = make_adapter([mocap_dead_end(), mocap_dead_end(), google_block(), google_block(), google_block(), None])

        adapter.repeatedly_attempt_to_join_meeting()

        self.assertEqual(blocked_messages_sent(adapter), [])
        self.assertEqual(adapter.attempt_to_join_meeting.call_count, 6)
        adapter.after_bot_joined_meeting.assert_called_once()

    def test_reports_blocked_by_platform_after_five_google_blocks(self):
        adapter = make_adapter([google_block() for _ in range(5)] + [None])

        adapter.repeatedly_attempt_to_join_meeting()

        self.assertEqual(len(blocked_messages_sent(adapter)), 1)
        self.assertEqual(adapter.attempt_to_join_meeting.call_count, 5)
        adapter.after_bot_joined_meeting.assert_not_called()

    def test_stops_retrying_when_mouse_dead_ends_never_resolve(self):
        allowance = web_bot_adapter.MAX_UNCOUNTED_EXPECTED_EXCEPTIONS
        adapter = make_adapter([mocap_dead_end() for _ in range(allowance + 5)] + [None])

        adapter.repeatedly_attempt_to_join_meeting()

        self.assertEqual(len(blocked_messages_sent(adapter)), 1)
        self.assertEqual(adapter.attempt_to_join_meeting.call_count, allowance + 5)


def make_controller():
    controller = BotController.__new__(BotController)
    controller.bot_in_db = MagicMock(id=42, join_at=None, created_at=timezone.now())
    controller.main_loop = MagicMock()
    controller.main_loop.is_running.return_value = True
    return controller


class TestBlockedByPlatformRestart(SimpleTestCase):
    @patch("bots.tasks.restart_bot_pod_task.restart_bot_pod.apply_async")
    @patch("bots.tasks.restart_bot_in_ephemeral_container_task.restart_bot_in_ephemeral_container.apply_async")
    def test_relaunches_through_the_docker_launcher_for_ephemeral_containers(self, docker_restart, pod_restart):
        controller = make_controller()

        with patch.dict("os.environ", {"LAUNCH_BOT_METHOD": "docker-compose-multi-host"}):
            controller.take_action_based_on_message_from_adapter(BLOCKED)

        docker_restart.assert_called_once()
        self.assertEqual(docker_restart.call_args.kwargs["args"], [42])
        self.assertEqual(docker_restart.call_args.kwargs["queue"], "bot_launcher_vm")
        pod_restart.assert_not_called()
        controller.main_loop.quit.assert_called_once()

    @patch("bots.tasks.restart_bot_pod_task.restart_bot_pod.apply_async")
    @patch("bots.tasks.restart_bot_in_ephemeral_container_task.restart_bot_in_ephemeral_container.apply_async")
    def test_keeps_recreating_the_pod_on_kubernetes(self, docker_restart, pod_restart):
        controller = make_controller()

        with patch.dict("os.environ", {"LAUNCH_BOT_METHOD": "kubernetes"}):
            controller.take_action_based_on_message_from_adapter(BLOCKED)

        pod_restart.assert_called_once_with(args=[42], countdown=60)
        docker_restart.assert_not_called()


def make_bot(last_event_type):
    bot = MagicMock(id=42, first_heartbeat_timestamp=123, last_heartbeat_timestamp=456)
    bot.ephemeral_container_name.return_value = "bot-42-bot-abc"
    last_event = MagicMock(event_type=last_event_type, requested_bot_action_taken_at=timezone.now(), metadata={})
    bot.last_bot_event.return_value = last_event
    return bot, last_event


@patch.object(restart_task.time, "sleep", MagicMock())
@patch.object(restart_task.run_bot_in_ephemeral_container, "apply_async")
@patch.object(restart_task.docker, "from_env")
@patch.object(restart_task, "Bot")
class TestRestartBotInEphemeralContainer(SimpleTestCase):
    def test_relaunches_a_bot_still_waiting_to_join_and_keeps_the_old_logs(self, Bot, from_env, launch):
        bot, last_event = make_bot(BotEventTypes.JOIN_REQUESTED)
        Bot.objects.get.return_value = bot
        old = MagicMock(status="exited")
        from_env.return_value.containers.get.return_value = old

        restart_bot_in_ephemeral_container.run(42)

        old.rename.assert_called_once()
        self.assertTrue(old.rename.call_args.args[0].startswith("bot-42-bot-abc-attempt-"))
        self.assertIsNone(last_event.requested_bot_action_taken_at)
        self.assertEqual(len(last_event.metadata["pod_recreations"]), 1)
        last_event.save.assert_called_once()
        self.assertIsNone(bot.first_heartbeat_timestamp)
        self.assertIsNone(bot.last_heartbeat_timestamp)
        bot.save.assert_called_once()
        launch.assert_called_once_with(args=[42], queue="bot_launcher_vm")

    def test_launches_straight_away_when_the_old_container_is_gone(self, Bot, from_env, launch):
        bot, _ = make_bot(BotEventTypes.JOIN_REQUESTED)
        Bot.objects.get.return_value = bot
        from_env.return_value.containers.get.side_effect = docker.errors.NotFound("gone")

        restart_bot_in_ephemeral_container.run(42)

        launch.assert_called_once_with(args=[42], queue="bot_launcher_vm")

    def test_skips_a_bot_that_is_no_longer_waiting_to_join(self, Bot, from_env, launch):
        bot, last_event = make_bot(BotEventTypes.LEAVE_REQUESTED)
        Bot.objects.get.return_value = bot

        restart_bot_in_ephemeral_container.run(42)

        launch.assert_not_called()
        from_env.return_value.containers.get.assert_not_called()
        last_event.save.assert_not_called()

    def test_removes_an_old_container_that_will_not_stop(self, Bot, from_env, launch):
        bot, _ = make_bot(BotEventTypes.JOIN_REQUESTED)
        Bot.objects.get.return_value = bot
        stuck = MagicMock(status="running")
        from_env.return_value.containers.get.return_value = stuck

        with patch.object(restart_task, "OLD_CONTAINER_EXIT_TIMEOUT_SECONDS", 0):
            restart_bot_in_ephemeral_container.run(42)

        stuck.remove.assert_called_once_with(force=True)
        stuck.rename.assert_not_called()
        launch.assert_called_once_with(args=[42], queue="bot_launcher_vm")
