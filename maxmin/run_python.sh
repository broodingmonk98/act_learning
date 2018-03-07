#!/bin/bash
#
echo $$ > /tmp/myscript.pid
#nohup python first.py tradeoff random  > output 2> error &

nohup python first.py tradeoff minmax  > output 2> error &

#nohup python first.py tradeoff minsum  > output 2> error &


#nohup python first.py tradeoff minmin  > output 2> error &

nohup python first.py tradeoff  max_query_vary_delta > output 2> error &

#nohup python first.py random  > output 2> error &
#nohup matlab -nodisplay -nojvm -nosplash -nodesktop -r " amtfl_main(10,'Reuters'); quit;" > output 2> error &
#nohup matlab -nodisplay -nojvm -nosplash -nodesktop -r " amtfl_main(5,'mnist'); quit;" > output 2> error &

