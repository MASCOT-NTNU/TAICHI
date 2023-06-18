# set the DNS server
echo "nameserver 1.1.1.1" > /tmp/resolv.conf

# add an IP address on the gateway's network:
add ip addr 192.168.1.199/24 dev eth0

# change the default gateway to use:
route add default gw 192.168.1.254 dev eth0
route del default gw 10.0.0.1

#You can now test the connection to the internet by using the command wget:
wget google.com


## Set up the network configuration
1. Update the nameserver in `/etc/resolv.conf` or `/tmp/resolv.conf`
2. Update the routing gateway. 
