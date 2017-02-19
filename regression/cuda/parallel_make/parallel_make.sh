#!/bin/bash

#configurations
regression_cuda="$HOME/repository/esbmc-gpu/regression/cuda/"

#erase temp files
if [ -e tests.log ]; then
	rm tests.log
fi
if [ -e hosts_set.tmp ]; then
	rm hosts_set.tmp
fi
if [ -e hosts_reset.tmp ]; then
	rm hosts_reset.tmp
fi

echo -n > command_list.pm

#create test_cases_list.pm
echo -n > test_cases_list.pm

for i in $(seq 2 $#)
do
	arg=`echo $* | cut -d" " -f$i`
	par=`echo $arg | cut -c${#arg}`
	if [ "$par" == "/" ]; then
		ls $regression_cuda$arg*/ -d >> test_cases_list.pm
	else
		echo "Error in args!"	
	fi
done
sed -e 's;'$HOME'/;\$HOME/;g' test_cases_list.pm > test_cases_list.tmp
mv test_cases_list.tmp test_cases_list.pm

#start execution
if [ -z $1 ];then
	echo "Provide the jobs number. Ex: ./make_parallel.sh 2"
else
	threads=$1
	./command_gen.sh
	INITIAL_EXECUTION_TIMESTAMP=$(date +%s)
	parallel -j$threads -a command_list.pm
	FINAL_EXECUTION_TIMESTAMP=$(date +%s)
fi

#ordering test cases result list
sort -n tests.log > tests.tmp
mv tests.tmp tests.log

#results
results_OK=`grep "\[Correct\]" tests.log | wc -l`
results_FAILED=`grep "\[Incorrect\]" tests.log | wc -l`
total_cases=`cat tests.log | wc -l`
echo "****************************" >> tests.log
echo "Correct results: $results_OK" >> tests.log
echo "Incorrect results: $results_FAILED" >> tests.log
#echo "Total test cases: $total_cases" >> tests.log
TIME=$((FINAL_EXECUTION_TIMESTAMP - INITIAL_EXECUTION_TIMESTAMP));
echo "Time(s): $TIME" >> tests.log
echo "****************************" >> tests.log

#hosts
#lvss01:lvss01:10.224.10.199:1:
#ceteli04:cuda04:10.224.10.166:1:
#ceteli03:ceteli2014:10.224.10.162:1:
#isabela:ceteli07:10.224.8.139:1:
#ceteli08:cuda08:10.224.9.9:1:
#maq01:maq01:10.224.1.162:1:
#maq02:maq02:10.224.1.123:1:
#maq03:maq03:10.224.1.121:1:
#maq04:maq04:10.224.1.120:1:
