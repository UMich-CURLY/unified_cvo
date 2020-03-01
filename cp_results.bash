prefix=$1

mkdir -p results/${prefix}/
for i in 00 01 02 03 04 05 06 07 08 09 10
do
	cp ${prefix}_${i}.txt results/${prefix}/${i}.txt

done
