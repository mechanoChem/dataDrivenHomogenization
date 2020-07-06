def parse_sys_args():
    import argparse, sys, os

    # check: https://docs.python.org/3.6/library/argparse.html#nargs
    parser = argparse.ArgumentParser(
        description='Run ML study for effective properties study',
        prog="'" + (sys.argv[0]) + "'")
    parser.add_argument(
        'configfile', help="configuration file for the study [*.config]")

    parser.add_argument(
        '-v', '--version', action='version', version='%(prog)s 0.1')
    #
    parser.add_argument(
        '-p',
        '--platform',
        choices=['cpu', 'gpu'],
        type=str,
        default='gpu',
        help='choose either use gpu or cpu platform (default: gpu)')

    #
    parser.add_argument(
        '-o', '--output_dir', type=str, help='folder name to store output data')
    parser.add_argument(
        '-r',
        '--restart_dir',
        type=str,
        help='folder name to store restart data')
    parser.add_argument(
        '-t',
        '--tensorboard_dir',
        type=str,
        help='folder name to store tensor board data')
    parser.add_argument(
        '-i',
        '--inspect',
        type=int,
        default=0,
        choices=[0, 1],
        help='pre-inspect the data (default: 0)')
    parser.add_argument(
        '-s',
        '--show',
        type=int,
        default=0,
        choices=[0, 1],
        help='show the final plot (default: 0)')

    #
    parser.add_argument(
        '-D',
        '--debug',
        type=bool,
        default=False,
        help="switch on/off the debug flag")
    parser.add_argument(
        '-V',
        '--verbose',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='verbose level of the code (default: 0)')
    parser.add_argument(
        '-P',
        '--profile',
        type=bool,
        default=False,
        help='switch on/off the profiling output')
    # parser.add_argument('--integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, help='sum the integers (default: find the max)') # for future references
    args = parser.parse_args()

    if (not (args.verbose == 3 and args.debug)):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress info output
        # tf.logging.set_verbosity(tf.logging.ERROR) # suppress deprecation warning
        #0 = all messages are logged (default behavior)
        #1 = INFO messages are not printed
        #2 = INFO and WARNING messages are not printed
        #3 = INFO, WARNING, and ERROR messages are not printed

    if (args.verbose == 3):
        print(parser.print_help())

    if (args.platform == 'cpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if (args.verbose == 3):
        ml_todos()

    return args


class sys_args:

    def __init__(self):
        self.configfile = ''
        self.platform = 'gpu'
        self.inspect = 0
        self.show = 0
        self.debug = False
        self.verbose = 0


def notebook_args(args):
    import sys, os

    if (not (args.verbose == 3 and args.debug)):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress info output
        # tf.logging.set_verbosity(tf.logging.ERROR) # suppress deprecation warning
        #0 = all messages are logged (default behavior)
        #1 = INFO messages are not printed
        #2 = INFO and WARNING messages are not printed
        #3 = INFO, WARNING, and ERROR messages are not printed

    if (args.platform == 'cpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if (args.verbose == 3):
        ml_todos()


if __name__ == "__main__":
    args = parse_sys_args()
    read_config_file(args.configfile)
