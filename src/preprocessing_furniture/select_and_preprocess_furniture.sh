#!/bin/bash

shuffle() {
   local i tmp size max rand

   # $RANDOM % (i+1) is biased because of the limited range of $RANDOM
   # Compensate by using a range which is a multiple of the array size.
   size=${#array[*]}
   max=$(( 32768 / size * size ))

   for ((i=size-1; i>0; i--)); do
      while (( (rand=$RANDOM) >= max )); do :; done
      rand=$(( rand % (i+1) ))
      tmp=${array[i]} array[i]=${array[rand]} array[rand]=$tmp
   done
}

furniture_path='Shelves'
CATEGORY_NAME='Shelf'
MAX_SIZE_Y=2.2
array=($(ls furniture_path/))
shuffle
i=0
num=0
while [ $num -lt 1 ]
do
	NAME=${array[$i]}
	file=furniture_path/$NAME

        ORIG_MAX_X=(`cat $file/models/model_normalized.json | jq '.max | .[0]'`)
        ORIG_MIN_X=(`cat $file/models/model_normalized.json | jq '.min | .[0]'`)
        ORIG_MAX_Y=(`cat $file/models/model_normalized.json | jq '.max | .[1]'`)
        ORIG_MIN_Y=(`cat $file/models/model_normalized.json | jq '.min | .[1]'`)

        ORIG_BB_X=$(echo "scale=6; $ORIG_MAX_X- $ORIG_MIN_X" | bc)
        ORIG_BB_Y=$(echo "scale=6; $ORIG_MAX_Y- $ORIG_MIN_Y" | bc)
        echo height: $ORIG_BB_Y
        if [[ $(echo "if (${ORIG_BB_Y} < ${MAX_SIZE_Y}) 1 else 0"| bc) -eq 1 ]]; then
                # Read the current bounding boxes
                NORMED_BBS=($(meshlabserver -i $file/models/model_normalized.obj -o test.obj -s filter.mlx | grep "Mesh Bounding Box Size"))
                # Calculate scale factor
                SCALE_FACTOR=$(echo "scale=6; $ORIG_BB_X/ ${NORMED_BBS[4]}" | bc)
                # Scale object
		SCALED='scaled.obj'
                cp scale.mlx current_scale.mlx
                sed -i "s/SCALE_VALUE/$SCALE_FACTOR/g" current_scale.mlx
                echo `meshlabserver -i $file/models/model_normalized.obj -o $file/models/model_orig.obj -s current_scale.mlx`
                # Simplify mesh and flip y-z axis
                echo `meshlabserver -i $file/models/model_orig.obj -o $SCALED -s simplify_meshes.mtl`
		# Move object to be sitting on the ground and y_max=0 (so we can easily place walls at y=0)
	        BBS_MIN=($(meshlabserver -i $SCALED -o test.obj -s filter.mlx | grep "Mesh Bounding Box min"))
	        BBS_MAX=($(meshlabserver -i $SCALED -o test.obj -s filter.mlx | grep "Mesh Bounding Box max"))
		echo ${BBS_MIN[@]}

	        # Move object
	        cp move.mlx current_move.mlx
	        sed -i "s/MOVE_Z_VALUE/$(echo "${BBS_MIN[6]} * -1" | bc)/g" current_move.mlx
	        sed -i "s/MOVE_Y_VALUE/$(echo "${BBS_MAX[5]} * -1" | bc)/g" current_move.mlx
	        echo `meshlabserver -i $SCALED -o processed_meshes_new/ShapeNet_${CATEGORY_NAME}_$NAME.obj -s current_move.mlx`
		num=$((num+1))
	else
		echo "Original bounding box y for $NAME was $ORIG_BB_Y"		

	fi
	((i++))

done
