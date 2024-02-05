#clean:
#    rm -r __pycache__
setup:  requirements.txt
	pip install -r requirements.txt

hello:
	echo "Hello, World"

image:
	mkdir .\image_banks\00_captured_raw  
	python -u "get_images.py"	

calib:
	python -u "camera_calibration.py"	
	

#clean_img:
#	rm ./image_banks/00_captured_raw/img2.png

clean_test:
	echo "Attempting to delete:"
	del .\image_banks\00_captured_raw\* /Q
	rmdir .\image_banks\00_captured_raw