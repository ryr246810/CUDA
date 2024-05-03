dynamic:
	cd build && cmake .. && make -j

clean:
	cd build && make clean && rm -rf *
	python3 Clean_Result.py 
	rm -f *~