from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument('-p', '--parallel', metavar='P', choices=['y', 'n'], type=str, 
                        default='n', help='parallel execution or not')
    parser.add_argument('-s', '--swarm-size', metavar='S', type=int, 
                        help='number of particles in the swarm', required=True)
    parser.add_argument('-i', '--iterations', metavar='I', type=int, 
                        help='number of iterations', required=True)
    parser.add_argument('-t', '--times', metavar='T', type=int, 
                        default=1, help='number of times to run')

    return parser.parse_args()
