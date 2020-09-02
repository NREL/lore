from datetime import datetime
from twisted.application import service
from twisted.application.internet import TimerService

# this will run the mediator as a service/daemon, see:
#   https://twistedmatrix.com/documents/current/core/howto/application.html
#   https://www.saltycrane.com/blog/2008/10/running-functions-periodically-using-twisteds-loopingcall/
#   http://www.christianlong.com/blog/twisted-on-windows-2015-edition.html
# this does not currently work

kUpdateInterval = 5     # [s]

def test_fun():
    print(datetime.now())

application = service.Application("lore")
timer_service = TimerService(kUpdateInterval, test_fun)
timer_service.setServiceParent(application)
