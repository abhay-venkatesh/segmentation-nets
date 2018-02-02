kill $(nvidia-smi | awk '$2=="Processes:" {p=1} p && $2 == 2 && $3 > 0 {print $3}')
