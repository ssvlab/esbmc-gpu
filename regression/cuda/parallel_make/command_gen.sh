#!/bin/bash

#generate command list
i=0
host_file_lines=`grep -cv '$=' hosts_file.config`
while [ $i -lt $host_file_lines ]
do
	#line of file
	j=`expr $i + 1`
	status[$i]=`cat hosts_file.config | head -$j | tail -1 | cut -d: -f4`
	if [ ${status[$i]} -eq 0 ]
	then
		users[$i]=`cat hosts_file.config | head -$j | tail -1 | cut -d: -f1`
		pass[$i]=`cat hosts_file.config | head -$j | tail -1 | cut -d: -f2`
		hosts[$i]=`cat hosts_file.config | head -$j | tail -1 | cut -d: -f3`
		
		#read and delete first test case of list
		path=`cat test_cases_list.pm | head -1`
		if [ "$path" == "" ]; then
			if [ -e hosts_set.tmp ]; then
				rm hosts_set.tmp
			fi
			exit
		fi
		sed -i 1d test_cases_list.pm

		#temporary out file for result. result will generate by result_gen.sh in tests.log
		result_temp=`echo $path | rev | cut -d"/" -f2 | rev`

		#create command line for parallel execution				
		IP_local=`ifconfig -a | grep inet | head -1 | awk '{print $2}' | cut -d: -f2`
		#localhost		
		if [ "$IP_local" == "${hosts[$i]}" ]; then
			command="flags=\`cat "$path"test.desc | head -2 | tail -1\` ; \$HOME/repository/esbmc-gpu/esbmc/esbmc "$path"main.cu -I ~/libraries --z3 \$flags > $result_temp ; ./lock_file.sh ; ./result_gen.sh $result_temp $path/test.desc ; ./reset_status.sh $j ; ./command_gen.sh ; ./unlock_file.sh"				
		else
			command="sshpass -p ${pass[$i]} ssh ${users[$i]}@${hosts[$i]} 'flags=\`cat "$path"test.desc | head -2 | tail -1\` ; \$HOME/repository/esbmc-gpu/esbmc/esbmc "$path"main.cu -I ~/libraries --z3 \$flags' > $result_temp ; ./lock_file.sh ; ./result_gen.sh $result_temp $path/test.desc ; ./reset_status.sh $j ; ./command_gen.sh ; ./unlock_file.sh"
		fi
		echo $command >> command_list.pm

		#set 1 status of host
		cat hosts_file.config | head -$j | tail -1 | sed -e 's/:0:/:1:/g' >> hosts_set.tmp
	else
		#no change status of host
		cat hosts_file.config | head -$j | tail -1 >> hosts_set.tmp
	fi
	i=`expr $i + 1`
done
mv hosts_set.tmp hosts_file.config
