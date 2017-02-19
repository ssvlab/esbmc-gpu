#!/bin/bash

if [ -e lock_file ];then
	while [ -e lock_file ];do
		sleep 0.1
	done
	touch lock_file
else
	touch lock_file
fi	
