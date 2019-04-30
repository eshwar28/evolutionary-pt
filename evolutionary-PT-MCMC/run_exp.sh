
#!/bin/sh 
echo Running all 	 
 
 

for prob in   3 4 5 6 7
	do
	python evo_pt_mcmc.py $prob 10 100 10  

done