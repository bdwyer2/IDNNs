"""
Train % plot networks in the information plane
"""
import idnns.networks.information_network


def main():
    print('Building the network')
    net = idnns.networks.information_network.informationNetwork()
    net.print_information()

    print('Start running the network')
    net.run_network()

    print('Saving data')
    net.save_data()

    print('Plotting figures')
    net.plot_network()


if __name__ == '__main__':
    main()

