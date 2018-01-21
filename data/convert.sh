I=0
for FILE in $(ls *.tif); do 
	I=$((I+1))
	convert -separate -format jpg $FILE $FILE-$I.jpg ; 
	
done ;
rename s/\.tif\-/\-/ *.jpg
