#!/bin/bash 指定批量删除

cat drop_database.txt | while read line
do 
echo "drop table $line;">>tt.txt
done 
tables='cat tt.txt'
hive -e "use hive_1;$tables"
# rm -f tt.txt

# #!/bin/sh
# hive -e "use hive_1; show tables;" | grep kylin_intermediate | while read line 
# do
# echo -n "drop table $line;" >> droptables.lst
# done
# tables=`cat droptables.lst`
# echo $tables
# hive -e "use hive_1; $tables"
# # rm droptables.lst
