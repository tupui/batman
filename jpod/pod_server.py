"""A command line interface to pod server."""
import sys
import os
import logging
import socket
from optparse import OptionParser
from pod import Pod, Snapshot
from misc import logging_conf, import_file
from rpyc.utils.server import ThreadedServer
import rpyc
import numpy as N
import mpi

# force numpy to raise an exception on floating-point errors
N.seterr(all='raise', under='warn')

help_message = '''
pod_server port script.py
'''


# setup logging
# pod_server is supposed to be run from the jpod output directory
# so we put the log file there
logging_conf.setup_file(os.getcwd(), 'poder')
logger = logging.getLogger('poder')


def map_reduce(expression, init=False):
    """Map reduce decorator.

    :param expression: expression to be executed on slave nodes
    :param init: process a class constructor
    """
    def wrap(method):
        def wrapped_method(self, *args, **kwargs):
            mpi.bcast((expression, args, kwargs))
            # local execution
            ret = method(self, *args, **kwargs)
            # get return values from all cpus
            ret = mpi.gather(ret)
            if not init:
                # filter object creation which must not return
                return ret
        return wrapped_method
    return wrap


class PodMPI(Pod):
    @map_reduce('pod = Pod(*args, **kwargs)', True)
    def __init__(self, *args, **kwargs):
        super(PodMPI, self).__init__(*args, **kwargs)

    @map_reduce('ret = pod.read(*args, **kwargs)')
    def read(self, *args, **kwargs):
        return super(PodMPI, self).read(*args, **kwargs)

    @map_reduce('ret = pod.write(*args, **kwargs)')
    def write(self, *args, **kwargs):
        return super(PodMPI, self).write(*args, **kwargs)

    @map_reduce('ret = pod.update(*args, **kwargs)')
    def update(self, *args, **kwargs):
        return super(PodMPI, self).update(*args, **kwargs)

    @map_reduce('ret = pod.decompose(*args, **kwargs)')
    def decompose(self, *args, **kwargs):
        return super(PodMPI, self).decompose(*args, **kwargs)

    @map_reduce('ret = pod.predict(*args, **kwargs)')
    def predict(self, *args, **kwargs):
        return super(PodMPI, self).predict(*args, **kwargs)


def run_server(port, settings):
    server = None

    if mpi.size == 1:
        class PodService(rpyc.Service):
            def on_disconnect(self):
                logger.info('Client is disconnected: closing server')
                server.close()
            class exposed_Pod(Pod):
                pass

        logger.info('Creating poder on %s:%i', socket.gethostname(), port)
        server = ThreadedServer(PodService, port=port, auto_register = False,
                                protocol_config={'allow_public_attrs' : True})
        server.start()

    else:
        if mpi.myid == 0:
            # Master serves, it holds a PodMPI instance as opposed to the slaves.
            class PodService(rpyc.Service):
                def on_disconnect(self):
                    logger.info('Client is disconnected: closing server')
                    server.close()
                    mpi.bcast(('return', None, None))

                class exposed_Pod(PodMPI):
                    pass

            logger.info('Creating poder on %s:%i', socket.gethostname(), port)
            server = ThreadedServer(PodService, port=port,
                                    auto_register = False,
                                    protocol_config={'allow_public_attrs' : True})
            server.start()

        else:
            # Slaves work with instances of Pod.
            #
            # Slaves scenario:
            #   * on the first loop occurence, they create a Pod object named 'pod',
            #   * then on subsequent loop occurences, they call a pod method and bind its return value to the variable 'ret'.
            #
            # In the body of the loop, a python expression is dynamically executed. The expression is formed of a stub and function's arguments. The stub contains a function name with a variable assignment (like 'ret = pod.dump'). The positional and optionnal arguments are passed to the named function in the stub.
            #
            # One loop occurence goes as follows:
            #   * wait for an expression from master node,
            #   * execute the expression
            #   * then return the expression's left hand side (if any) to master.

            # local namespace scope for using exec,
            # it keeps track of the already created variables over multiple loop iterations
            scope = {}
            while True:
                (expression, args, kwargs) = mpi.bcast()
                # logger.info(expression)
                # logger.info(args)
                # logger.info(kwargs)
                if expression == 'return':
                    # TODO: how to tell exec it's in a function?
                    return
                scope.update(locals())
                exec expression in globals(), scope
                mpi.gather(scope.get('ret'))




def main(argv=None):
    """Parse and check options, and then call XXX()."""

    if argv is None:
        argv = sys.argv #[1:]

    parser = OptionParser(usage=help_message)

    # parse command line
    (options, args) = parser.parse_args()

    if len(args) != 2:
        parser.error("incorrect number of arguments")

    port     = args[0]
    settings = import_file(args[1])

    try:
        run_server(int(port), settings)
        return 0
    except:
        logger.exception('Exception caught on cpu %i'%mpi.myid)
        return 1


if __name__ == "__main__":
    sys.exit(main())
