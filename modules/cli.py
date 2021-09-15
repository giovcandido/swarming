from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument('-p', '--parallel', help="parallel execution or not", required=True)
    parser.add_argument('-s', '--swarm-size', help="number of particles in the swarm", required=True)
    parser.add_argument('-m', '--max-iterations', help="maximum number of iterations", required=True)
    parser.add_argument('-t', '--times', help="number of times to run", required=True)

    return parser.parse_args()
