array=(
  Vietnam
  Germany
  Argentina
)
array2=(
  Asia
  Europe
  America
)

for index in ${!array[*]}; do 
  echo "${array[$index]} is in ${array2[$index]}"
done
