PYTHONPATH=$(CURDIR)

#clean:
#    rm -r __pycache__
setup:  requirements.txt
	pip install -r requirements.txt

image:
	mkdir .\image_banks\00_captured_raw 
	python -u "get_images.py"	

calib:
	mkdir .\image_banks\01_with_chess 
	mkdir .\image_banks\02_undistorted 
	python -u "camera_calibration.py"	
	
pose:
	mkdir .\image_banks\03_posed 
	python -u "pose_comp.py"	

clean:
	echo "Attempting to delete all image banks:"
	rmdir .\image_banks /s /q
