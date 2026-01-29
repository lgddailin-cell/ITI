#!/bin/sh
sed -i '/:1000:/d' /etc/group
sed -i '/:1000:/d' /etc/passwd
echo domainusers:x:20000513:domainusers >> /etc/group
echo 25171213997:x:13004:20000513:25171213997:/home/25171213997:/bin/sh >> /etc/passwd
echo 25171213997:\*:99999:0:99999:7::: >> /etc/shadow
if [ -f /etc/sudoers ]; then
         echo '25171213997 ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers
         fi
cd "/home/25171213997/ITI/codes/"
su - 25171213997 -c 'cd "/home/25171213997/ITI/codes/";/home/25171213997/ITI/codes/cuto.sh'
