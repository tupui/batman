"""Set logging configuration"""
import logging
import os
import mpi

full_format = '[' + str(mpi.myid) + '] %(asctime)s  %(name)s\n\t%(message)s'
light_format = '[' + str(mpi.myid) + '] %(name)s\n\t%(message)s'
datefmt = '%x %H:%M:%S'


def setup_file(path, name):
    file_path = os.path.join(path, '%s.%i.log'%(name, mpi.myid))

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format=full_format,
                        datefmt=datefmt,
                        filename=file_path,
                        filemode='w')


def setup(path, name):
    setup_file(path, name)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(light_format)
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
