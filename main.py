"""
Train % plot networks in the information plane
"""
from idnns.networks import information_network as inet


def main():
    # Build the network
    print('Building the network')
    net = inet.informationNetwork()
    net.print_information()
    print('Start running the network')
    net.run_network()
    print('Saving data')
    net.save_data()
    print('Plotting figures')
    # Plot the network
    net.plot_network()


if __name__ == '__main__':
    main()

