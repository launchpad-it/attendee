class UiException(Exception):
    def __init__(self, message, step, inner_exception):
        self.step = step
        self.inner_exception = inner_exception
        super().__init__(message)


# When this exception is raised, the bot will stop running and log that it was denied access to the meeting
class UiRequestToJoinDeniedException(UiException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


# When this exception is raised, the bot will stop running and log that it could not join the meeting because it waited too long for the host to start the meeting
class UiCouldNotJoinMeetingWaitingForHostException(UiException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


# When this exception is raised, the bot will stop running and log that it could not join the meeting because it waited too long in the waiting room
class UiCouldNotJoinMeetingWaitingRoomTimeoutException(UiException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


# When this exception is raised, the bot will stop running and log that the meeting was not found
class UiMeetingNotFoundException(UiException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


class UiIncorrectPasswordException(UiException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


class UiBlockedByCaptchaException(UiException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


class UiRetryableException(UiException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


class UiAuthorizedUserNotInMeetingTimeoutExceededException(UiException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


class UiRetryableExpectedException(UiRetryableException):
    # Whether this failure is evidence that the platform is blocking the bot. Only
    # such failures count toward BLOCKED_BY_PLATFORM_REPEATEDLY in
    # WebBotAdapter.repeatedly_attempt_to_join_meeting; the others get a bounded
    # free allowance first (MAX_UNCOUNTED_EXPECTED_EXCEPTIONS).
    indicates_platform_blocking = True

    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


# When this exception is raised, the bot will always retry
# It is up to the adapter to stop throwing this exception eventually
class UiInfinitelyRetryableException(UiException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


class UiCouldNotLocateElementException(UiRetryableException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


class UiCouldNotClickElementException(UiRetryableException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


class UiLoginRequiredException(UiException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)


class UiLoginAttemptFailedException(UiException):
    def __init__(self, message, step=None, inner_exception=None):
        super().__init__(message, step, inner_exception)
