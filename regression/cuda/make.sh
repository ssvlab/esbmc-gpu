#!/bin/bash

#set print error e/or time
while getopts ethcs: OPTION; do
	case "${OPTION}" in
		e) PRINT_ERROR=1 ;;
		t) PRINT_TIME=1 ;;
		h) PRINT_HELP=1;;
		c) GEN_CSV_FILE=1;;
		s) USER_SOLVER="${OPTARG}";;
	esac
done

#definitions
LOG_OUT_DEFAULT="tests.log"
CSV_OUT_DEFAULT="tests.csv"

if [ $USER_SOLVER ]; then
	SOLVER=$USER_SOLVER
	echo "::::Using solver $SOLVER::::"
else
	SOLVER="z3"
	echo "::::Using solver default $SOLVER::::"
fi


########HELP###########
if [ ${PRINT_HELP} ]; then
echo -e "\033[33m"
echo "Este script executa o ESBMC sobre casos de teste CUDA em"
echo "pastas e subpastas que possuem o arquivo main.cu."
echo ""
echo "#Configurações:"
echo ""
echo "-ESBMC"
echo "O script usa a versão do ESBMC definida na variável de"
echo "ambiente do Linux."
echo ""
echo "-Bibliotecas"
echo "\$HOME/libraries"
echo ""
echo "-test.desc"
echo "Caso de Teste"
echo "sem falha:			com falha:"
echo "main.cu				main.cu"
echo "flags de execução		flags de execução"
echo "^VERIFICATION SUCCESSFUL$	^VERIFICATION FAILED$"
echo "#VERIFICATION SUCCESSFUL$	#motivo da falha$ (e.g., #array bounds violated$)"
echo ""
echo "->Motivos de falha:"
echo "	#Access to object out of bounds$ : acesso fora dos limites de um objeto"
echo "	#array bounds violated$ : acesso fora dos limites de um vetor"
echo "	#assertion$ : 		assertiva incorreta"
echo "	#constant memory$ : 	escrita em memória constante"
echo "	#NULL pointer$ : 	acesso a ponteiro nulo"
echo "	#R/W data race$ : 	condição de corrida R/W"
echo "	#W/W data race$ : 	condição de corrida W/W"
echo ""
echo "#Uso:"
echo ""
echo "./make.sh		exibe o resultado padrão (e.g., cuda01 [Correct])"
echo ""
echo "#Opções:"
echo ""
echo "-c,			salva a saída em um arquivo de vírgulas (.csv)"
echo "-e,			exibe o motivo de falha do caso de teste"
echo "-h,			exibe o texto de ajuda"
echo "-s[solver_name],	seleciona um solver (padrão: Z3)"
echo "-t,			exibe o tempo de execução para cada caso de teste em minutos"
echo ""
echo "O resumo final apresenta a quantidade de casos corretos e"
echo "incorretos, e o tempo total de execução em segundos."
echo ""
echo "Make script versão 2.0 - 11/2015"
echo "Desenvolvido por Phillipe Pereira"
echo "Email: apphillipe@gmail.com"
echo -e "\033[0m"
exit
fi
######################

########FUNCTION FOR PRINT RESULTS###########
function generate_csv(){
	if [ ${GEN_CSV_FILE} ]; then
		if [ -e "$CSV_OUT_DEFAULT" ]; then
			echo "$1,$2,$3,$4,$5" >> "$CSV_OUT_DEFAULT"
		else
			echo "Arquivo,Resultado esperado,Resultado,Saída,Tempo(m)" >> "$CSV_OUT_DEFAULT"
			echo "$1,$2,$3,$4,$5" >> "$CSV_OUT_DEFAULT"
		fi
	fi
}
######################

########FUNCTION FOR PRINT RESULTS###########
function print_result(){
	if [ ${PRINT_TIME} ]; then
		echo "$1 [$2] [time(m): $3]" >> "$LOG_OUT_DEFAULT"
		echo "$1 [$2] [time(m): $3]"
	else
		echo "$1 [$2]" >> "$LOG_OUT_DEFAULT"
		echo "$1 [$2]"
	fi
}
######################

########FUNCTION GENERATE RESULTS###########
#generate result file ex: cuda01 [correct]
function result_gen(){
FAILED=`cat $1 | tail -1`
SUCCESS=`cat $1 | tail -2 | head -1`

#report for incorrect in violated property or successful
REPORT_LINE=`cat $1 | grep -n "Violated property" | cut -d':' -f1`
LINES=`cat $1 | wc -l`
BLOCK=`expr $LINES - $REPORT_LINE + 1`
BLOCK_VIOLATED=`cat $1 | tail -$BLOCK`

#define if either correct or incorrect
EXPECTED_RESULT=`cat $2 | tail -2 | head -1`
EXPECTED_RESULT=${EXPECTED_RESULT/$/}
EXPECTED_RESULT=${EXPECTED_RESULT/^/}

#define the error type
ERROR_TYPE=`cat $2 | tail -1`
ERROR_TYPE=${ERROR_TYPE/$/}
ERROR_TYPE=${ERROR_TYPE/\#/}

#witness report
WIT_REPORT=`echo $BLOCK_VIOLATED | grep "$ERROR_TYPE" | wc -l`

#parsing error report
PARSING_ERROR=`echo $BLOCK_VIOLATED | grep "PARSING ERROR" | wc -l`

#conversion error report
CONVERSION_ERROR=`echo $BLOCK_VIOLATED | grep "CONVERSION ERROR" | wc -l`

#time out report
TIME_OUT_ERROR=`echo $BLOCK_VIOLATED | grep "Timed out" | wc -l`

NAME_FILE=`echo $1 | sed 's/_tmp//'`
if [ "$FAILED" == "$EXPECTED_RESULT" ]; then
	#save in $LOG_OUT_DEFAULT; print on view
	if [ $WIT_REPORT -ne 0 ]; then	
		print_result $NAME_FILE "False Correct" $3
		generate_csv "$NAME_FILE" "$ERROR_TYPE" "False Correct" "$ERROR_TYPE" "$3"
	else
		print_result $NAME_FILE "False Incorrect" $3
		generate_csv "$NAME_FILE" "$ERROR_TYPE" "False Incorrect" "\"$BLOCK_VIOLATED\"" "$3"
	fi

	#correct fail reason
	BLOCK=`expr $BLOCK + 1`
	if [ ${PRINT_ERROR} ]; then
		if [ $BLOCK -eq $LINES ];
		then
			echo $FAILED >> $LOG_OUT_DEFAULT
		else
			echo $BLOCK_VIOLATED >> $LOG_OUT_DEFAULT
		fi
	fi

elif [ "$SUCCESS" == "$EXPECTED_RESULT" ];then
	#save in $LOG_OUT_DEFAULT; print on view
	print_result $NAME_FILE "True Correct" $3
	generate_csv "$NAME_FILE" "Success" "True Correct" "Success" "$3"
else
	#save in $LOG_OUT_DEFAULT; print on view
	if [ "$FAILED" == "VERIFICATION FAILED" ]; then
		print_result $NAME_FILE "False Incorrect" $3
		generate_csv "$NAME_FILE" "Success" "False Incorrect" "\"$BLOCK_VIOLATED\"" "$3"

	elif [ "$SUCCESS" == "VERIFICATION SUCCESSFUL" ]; then
		if [ $TIME_OUT_ERROR -ne 0 ]; then
			print_result $NAME_FILE "Time out" $3
			generate_csv "$NAME_FILE" "-" "Time out" "-" "$3" 
		else
			print_result $NAME_FILE "True Incorrect" $3
			generate_csv "$NAME_FILE" "$ERROR_TYPE" "True Incorrect" "Success" "$3"
		fi

	elif [ $PARSING_ERROR -ne 0 ]; then
		print_result $NAME_FILE "Not Supported" $3
		generate_csv "$NAME_FILE" "-" "Not Supported" "-" "$3"

	elif [ $CONVERSION_ERROR -ne 0 ]; then
		print_result $NAME_FILE "Not Supported" $3
		generate_csv "$NAME_FILE" "-" "Not Supported" "-" "$3"

	elif [ $TIME_OUT_ERROR -ne 0 ]; then
		print_result $NAME_FILE "Time out" $3
		generate_csv "$NAME_FILE" "-" "Time out" "-" "$3"
	
	else 
		print_result $NAME_FILE "Error" $3
		generate_csv "$NAME_FILE" "-" "Error" "-" "$3"
	fi	

	#incorrect reason
	BLOCK=`expr $BLOCK + 1`
	if [ ${PRINT_ERROR} ]; then
		if [ $BLOCK -eq $LINES ];
		then
			echo $SUCCESS >> $LOG_OUT_DEFAULT	
		else
			echo $BLOCK_VIOLATED >> $LOG_OUT_DEFAULT
		fi
	fi
fi

#delete temporary result file
rm $1
}
######################

########EXECUTE###########
#erase temp files
if [ -e "$LOG_OUT_DEFAULT" ]; then
	rm $LOG_OUT_DEFAULT
fi
if [ ${GEN_CSV_FILE} ]; then
	if [ -e "$CSV_OUT_DEFAULT" ]; then
		rm $CSV_OUT_DEFAULT
	fi
fi

#generate test cases list
find * -name main.cu | sort | sed 's:main.cu::' > test_cases_list.pm

#start execution
line=0
tcl_lines=`grep -cv '$=' test_cases_list.pm`

INITIAL_EXECUTION_TIMESTAMP=$(date +%s)
while [ $line -lt $tcl_lines ]
do
	path=`cat test_cases_list.pm | head -1`
	if [ "$path" == "" ]; then
		exit
	fi
	sed -i 1d test_cases_list.pm
	
	#temporary out file to results. Result generated by result_gen() in $LOG_OUT_DEFAULT
	result_temp=`echo $path | rev | cut -d"/" -f2 | rev `
	result_temp=`echo $result_temp"_tmp"`
	flags=`cat "$path"test.desc | head -2 | tail -1` 

	/usr/bin/time -otimetmp esbmc "$path"main.cu -I ~/libraries --"$SOLVER" --timeout 900 $flags &> $result_temp

	TIME=`head -1 timetmp | cut -d" " -f3 | sed 's/elapsed//'`
	result_gen $result_temp $path/test.desc $TIME
	line=`expr $line + 1`
done
FINAL_EXECUTION_TIMESTAMP=$(date +%s)

#remove tmp files
rm timetmp test_cases_list.pm
###############################

#save and print results
TRUE_CORRECT=`grep "\[True Correct\]" "$LOG_OUT_DEFAULT" | wc -l`
TRUE_INCORRECT=`grep "\[True Incorrect\]" "$LOG_OUT_DEFAULT" | wc -l`
FALSE_CORRECT=`grep "\[False Correct\]" "$LOG_OUT_DEFAULT" | wc -l`
FALSE_INCORRECT=`grep "\[False Incorrect\]" "$LOG_OUT_DEFAULT" | wc -l`
TIME_OUT_RESULT=`grep "\[Time out\]" "$LOG_OUT_DEFAULT" | wc -l`
NOT_SUPPORTED=`grep "\[Not Supported\]" "$LOG_OUT_DEFAULT" | wc -l`
ERROR_RESULT=`grep "\[Error\]" "$LOG_OUT_DEFAULT" | wc -l`
TOTAL=`expr $TRUE_CORRECT + $TRUE_INCORRECT + $FALSE_CORRECT + $FALSE_INCORRECT + $NOT_SUPPORTED + $ERROR_RESULT + $TIME_OUT_RESULT`
ALLTIME=$((FINAL_EXECUTION_TIMESTAMP - INITIAL_EXECUTION_TIMESTAMP));

echo "****************************" >> $LOG_OUT_DEFAULT
echo "Solver: $SOLVER" >> $LOG_OUT_DEFAULT
echo "True Correct: $TRUE_CORRECT" >> $LOG_OUT_DEFAULT
echo "False Correct: $FALSE_CORRECT" >> $LOG_OUT_DEFAULT
echo "True Incorrect: $TRUE_INCORRECT" >> $LOG_OUT_DEFAULT
echo "False Incorrect: $FALSE_INCORRECT" >> $LOG_OUT_DEFAULT
echo "Time out: $TIME_OUT_RESULT" >> $LOG_OUT_DEFAULT
echo "Not Supported: $NOT_SUPPORTED" >> $LOG_OUT_DEFAULT
echo "Error: $ERROR_RESULT" >> $LOG_OUT_DEFAULT
echo "Total: $TOTAL" >> $LOG_OUT_DEFAULT
echo "Time(s): $ALLTIME" >> $LOG_OUT_DEFAULT
echo "****************************" >> $LOG_OUT_DEFAULT

echo "****************************"
echo "Solver: $SOLVER"
echo "True Correct: $TRUE_CORRECT"
echo "False Correct: $FALSE_CORRECT"
echo "True Incorrect: $TRUE_INCORRECT"
echo "False Incorrect: $FALSE_INCORRECT"
echo "Time out: $TIME_OUT_RESULT"
echo "Not Supported: $NOT_SUPPORTED"
echo "Error: $ERROR_RESULT"
echo "Total: $TOTAL"
echo "Time(s): $ALLTIME"
echo "****************************"

if [ ${GEN_CSV_FILE} ]; then
	echo "****************************"  >> "$CSV_OUT_DEFAULT"
	echo "Solver: $SOLVER"  >> "$CSV_OUT_DEFAULT"
	echo "True Correct: $TRUE_CORRECT"  >> "$CSV_OUT_DEFAULT"
	echo "False Correct: $FALSE_CORRECT"  >> "$CSV_OUT_DEFAULT"
	echo "True Incorrect: $TRUE_INCORRECT"  >> "$CSV_OUT_DEFAULT"
	echo "False Incorrect: $FALSE_INCORRECT"  >> "$CSV_OUT_DEFAULT"
	echo "Time out: $TIME_OUT_RESULT"  >> "$CSV_OUT_DEFAULT"
	echo "Not Supported: $NOT_SUPPORTED"  >> "$CSV_OUT_DEFAULT"
	echo "Error: $ERROR_RESULT"  >> "$CSV_OUT_DEFAULT"
	echo "Total: $TOTAL"  >> "$CSV_OUT_DEFAULT"
	echo "Time(s): $ALLTIME"  >> "$CSV_OUT_DEFAULT"
	echo "****************************"  >> "$CSV_OUT_DEFAULT"
fi
