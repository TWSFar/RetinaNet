pst_thd=0.05
add=0.01
for i in `seq 1 60`
do
	pst_thd=$(echo "$pst_thd + $add" | bc)
    python train.py eval --batch_size 8 --pst_thd $pst_thd
done

echo $pst_thd