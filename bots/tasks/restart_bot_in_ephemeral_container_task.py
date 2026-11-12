import logging
import time

import docker
from celery import shared_task

from bots.models import Bot, BotEventTypes

from .run_bot_in_ephemeral_container_task import run_bot_in_ephemeral_container

logger = logging.getLogger(__name__)

# How long to wait for the previous container to exit on its own. The bot quits its
# main loop right after asking for a restart, so this normally takes a second or two.
OLD_CONTAINER_EXIT_TIMEOUT_SECONDS = 60


def _set_aside_previous_container(client, container_name):
    """Free the bot's container name for the relaunch.

    An exited container is renamed rather than removed, so its logs stay available
    when BOT_CONTAINER_AUTO_REMOVE=false. One that is still running after the
    timeout is removed, as restart_bot_pod deletes a lingering pod.
    """
    try:
        container = client.containers.get(container_name)
    except docker.errors.NotFound:
        return

    deadline = time.monotonic() + OLD_CONTAINER_EXIT_TIMEOUT_SECONDS
    container.reload()
    while container.status == "running" and time.monotonic() < deadline:
        time.sleep(2)
        container.reload()

    if container.status == "running":
        logger.warning(f"Container {container_name} still running after {OLD_CONTAINER_EXIT_TIMEOUT_SECONDS}s, removing it before relaunch")
        container.remove(force=True)
        return

    new_name = f"{container_name}-attempt-{int(time.time())}"
    container.rename(new_name)
    logger.info(f"Renamed previous container {container_name} to {new_name}")


@shared_task(bind=True, soft_time_limit=600)
def restart_bot_in_ephemeral_container(self, bot_id):
    """Docker counterpart of restart_bot_pod, for LAUNCH_BOT_METHOD=docker-compose-multi-host.

    Relaunches a bot that gave up joining because the platform kept blocking it.
    Must run on the bot_launcher_vm queue: only the launcher has the Docker socket.
    """
    logger.info(f"Restarting ephemeral container for bot {bot_id}")

    bot = Bot.objects.get(id=bot_id)
    last_bot_event = bot.last_bot_event()
    if last_bot_event is None or last_bot_event.event_type != BotEventTypes.JOIN_REQUESTED:
        logger.info(f"Bot {bot_id} is not in JOINING state, so not restarting its container")
        return {"bot_id": bot_id, "status": "skipped"}

    _set_aside_previous_container(docker.from_env(), bot.ephemeral_container_name())

    # Same bookkeeping as restart_bot_pod: the new run must be allowed to act on the
    # join request again, and stale heartbeats must not get it reaped.
    last_bot_event.requested_bot_action_taken_at = None
    if not last_bot_event.metadata:
        last_bot_event.metadata = {}
    last_bot_event.metadata.setdefault("pod_recreations", []).append(int(time.time()))
    last_bot_event.save()

    bot.first_heartbeat_timestamp = None
    bot.last_heartbeat_timestamp = None
    bot.save()

    run_bot_in_ephemeral_container.apply_async(args=[bot_id], queue="bot_launcher_vm")
    return {"bot_id": bot_id, "status": "relaunched"}
