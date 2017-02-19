#!/bin/bash

#generate result file ex: cuda01 [correct]
failed=`cat $1 | tail -1`
success=`cat $1 | tail -2 | head -1`

#report for incorrect in violated property or successful
report_line=`cat $1 | grep -n "Violated property" | cut -d':' -f1`
lines=`cat $1 | wc -l`
block=`expr $lines - $report_line + 1`
violated=`cat $1 | tail -$block`

#define if is either correct or incorrect
result=`cat $2 | tail -1`
result=${result/$/}
result=${result/^/}

if [ "$failed" == "$result" ]; then
	echo $1 [Correct] >> tests.log
	print="[Correct]"
	#correct fail reason
	block=`expr $block + 1`
	if [ $block -eq $lines ];
	then
		echo $failed >> tests.log	
	else
		echo $violated >> tests.log
	fi

elif [ "$success" == "$result" ];then
	echo $1 [Correct] >> tests.log
	print="[Correct]"
else
	echo $1 [Incorrect] >> tests.log
	print="[Incorrect]"

	#incorrect reason
	block=`expr $block + 1`
	if [ $block -eq $lines ];
	then
		echo $success >> tests.log	
	else
		echo $violated >> tests.log
	fi
fi

#print result on RT
echo $1 $print

#delete temporary result file
rm $1
