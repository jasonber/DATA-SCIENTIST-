卸载： sudo apt-get remove mysql-common
卸载：sudo apt-get autoremove --purge mysql-server-5.7
清除残留数据：dpkg -l|grep ^rc|awk '{print$2}'|sudo xargs dpkg -P
再次查看MySQL的剩余依赖项：dpkg --list|grep mysql
继续删除剩余依赖项，如：sudo apt-get autoremove --purge mysql-apt-config
