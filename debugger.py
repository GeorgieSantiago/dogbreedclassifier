import logging

"""
DEBUG
INFO
WARNING
ERROR
CRITICAL
"""
# asctime: time of the log was printed out
# levelname: name of the log
# datefmt: format the time of the log
# give DEBUG log
rfh = logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.DEBUG,
    filename='logs.log')

logger = logging.getLogger('my_app')
