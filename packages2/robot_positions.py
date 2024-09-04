import math

class RobotPositions:
    vacuum_gripper = 9

    home = {
        "base": math.radians(0),
        "shoulder": math.radians(-90),
        "elbow": math.radians(0),
        "wrist1": math.radians(-90),
        "wrist2": math.radians(0),
        "wrist3": math.radians(0)                   # home position
    }


    pose_wait_base = {
        "base": math.radians(-45),
        "shoulder": math.radians(-112),
        "elbow": math.radians(100),
        "wrist1": math.radians(-80),
        "wrist2": math.radians(-90),
        "wrist3": math.radians(0)                   # base position
    }



    look_at_chessboard = {
        "base": math.radians(-58),
        "shoulder": math.radians(-108),
        "elbow": math.radians(54),
        "wrist1": math.radians(-36),
        "wrist2": math.radians(-92),
        "wrist3": math.radians(-16)                   # position
    }

    calib_pose_1 = {
        "base": math.radians(-62),
        "shoulder": math.radians(-86),
        "elbow": math.radians(48.5),
        "wrist1": math.radians(-36),
        "wrist2": math.radians(-92),
        "wrist3": math.radians(-16)                   # position ok
    }  
    calib_pose_2 = {
        "base": math.radians(-42.2),
        "shoulder": math.radians(-100),
        "elbow": math.radians(47.5),
        "wrist1": math.radians(-34.5),
        "wrist2": math.radians(-101.5),
        "wrist3": math.radians(-30)                   # position ok
    }  
    calib_pose_3 = {
        "base": math.radians(-42.2),
        "shoulder": math.radians(-129),
        "elbow": math.radians(84.5),
        "wrist1": math.radians(-53),
        "wrist2": math.radians(-96.2),
        "wrist3": math.radians(-3)                   # position ok
    }  
    calib_pose_4 = {
        "base": math.radians(-22.5),
        "shoulder": math.radians(-129),
        "elbow": math.radians(83),
        "wrist1": math.radians(-55),
        "wrist2": math.radians(-96),
        "wrist3": math.radians(33)                   # position ok
    }  
    calib_pose_5 = {
        "base": math.radians(-21),
        "shoulder": math.radians(-139),
        "elbow": math.radians(85.5),
        "wrist1": math.radians(-55),
        "wrist2": math.radians(-96.2),
        "wrist3": math.radians(-32)                   # position ok
    }  
    calib_pose_6 = {
        "base": math.radians(-60),
        "shoulder": math.radians(-150),
        "elbow": math.radians(89),
        "wrist1": math.radians(-49),
        "wrist2": math.radians(-96),
        "wrist3": math.radians(-24)                   # position ok
    }  
    calib_pose_7 = {
        "base": math.radians(-49),
        "shoulder": math.radians(-116),
        "elbow": math.radians(76),
        "wrist1": math.radians(-52),
        "wrist2": math.radians(-97),
        "wrist3": math.radians(-7)                   # position ok
    }  
    calib_pose_8 = {
        "base": math.radians(-49),
        "shoulder": math.radians(-85.5),
        "elbow": math.radians(38),
        "wrist1": math.radians(-28),
        "wrist2": math.radians(-98),
        "wrist3": math.radians(-7)                   # position ok
    }  
    calib_pose_9 = {
        "base": math.radians(-52),
        "shoulder": math.radians(-91),
        "elbow": math.radians(18.5),
        "wrist1": math.radians(-19),
        "wrist2": math.radians(-98),
        "wrist3": math.radians(-14)                   # position ok
    }  
    calib_pose_10 = {
        "base": math.radians(-52),
        "shoulder": math.radians(-150),
        "elbow": math.radians(112.5),
        "wrist1": math.radians(-69),
        "wrist2": math.radians(-96),
        "wrist3": math.radians(-10)                   # position ok
    }  

    poses = [
        calib_pose_1,
        calib_pose_2,
        calib_pose_3,
        calib_pose_4,
        # calib_pose_5,
        calib_pose_6,
        calib_pose_7,
        calib_pose_8,
        calib_pose_9,
        calib_pose_10
    ]



    calib_2_pose_1 = {
            "base": math.radians(-57.57),
            "shoulder": math.radians(-111.91),
            "elbow": math.radians(57.95),
            "wrist1": math.radians(-39.91),
            "wrist2": math.radians(-91.97),
            "wrist3": math.radians(-16)                   # position ok
        }  

    calib_2_pose_2 = {
            "base": math.radians(-57.57),
            "shoulder": math.radians(-111.94),
            "elbow": math.radians(64.11),
            "wrist1": math.radians(-43.59),
            "wrist2": math.radians(-91.63),
            "wrist3": math.radians(-16)                   # position ok
        }  

    calib_2_pose_3 = {
            "base": math.radians(-57.57),
            "shoulder": math.radians(-114.26),
            "elbow": math.radians(74.08),
            "wrist1": math.radians(-48.38),
            "wrist2": math.radians(-89.67),
            "wrist3": math.radians(-16)                   # position ok
        }  


    calib_2_pose_4 = {
            "base": math.radians(-57.57),
            "shoulder": math.radians(-118.37),
            "elbow": math.radians(75.75),
            "wrist1": math.radians(-47.04),
            "wrist2": math.radians(-88.97),
            "wrist3": math.radians(-16)                   # position ok
        }  


    calib_2_pose_5 = {
            "base": math.radians(-52.68),
            "shoulder": math.radians(-125.02),
            "elbow": math.radians(76.11),
            "wrist1": math.radians(-44.81),
            "wrist2": math.radians(-89.02),
            "wrist3": math.radians(-10.06)                   # position ok
        }  



    calib_2_pose_6 = {
            "base": math.radians(-52.67),
            "shoulder": math.radians(-124.88),
            "elbow": math.radians(72.67),
            "wrist1": math.radians(-43.47),
            "wrist2": math.radians(-87.69),
            "wrist3": math.radians(-10.06)                   # position ok
        }  


    calib_2_pose_7 = {
            "base": math.radians(-69.76),
            "shoulder": math.radians(-124.73),
            "elbow": math.radians(72.88),
            "wrist1": math.radians(-41.26),
            "wrist2": math.radians(-87.82),
            "wrist3": math.radians(-27.89)                   # position ok
        }  


    calib_2_pose_8 = {
            "base": math.radians(-69.76),
            "shoulder": math.radians(-124.76),
            "elbow": math.radians(68.99),
            "wrist1": math.radians(-38.11),
            "wrist2": math.radians(-87.74),
            "wrist3": math.radians(-27.89)                   # position ok
        }  


    calib_2_pose_9 = {
            "base": math.radians(-69.75),
            "shoulder": math.radians(-116.17),
            "elbow": math.radians(69.06),
            "wrist1": math.radians(-39.70),
            "wrist2": math.radians(-87.83),
            "wrist3": math.radians(-27.89)                   # position ok
        }  


    calib_2_pose_10 = {
            "base": math.radians(-61.08),
            "shoulder": math.radians(-114.92),
            "elbow": math.radians(71.42),
            "wrist1": math.radians(-43.64),
            "wrist2": math.radians(-87.82),
            "wrist3": math.radians(-19.57)                   # position ok
        }  

    calib_2_pose_11 = {
            "base": math.radians(-98.92),
            "shoulder": math.radians(-104.81),
            "elbow": math.radians(54.06),
            "wrist1": math.radians(-36.36),
            "wrist2": math.radians(-87.73),
            "wrist3": math.radians(-56.47)                   # position ok
        }  


    calib_2_pose_12 = {
            "base": math.radians(-98.75),
            "shoulder": math.radians(-119.07),
            "elbow": math.radians(55.08),
            "wrist1": math.radians(-28.18),
            "wrist2": math.radians(-87.87),
            "wrist3": math.radians(-56.47)                   # position ok
        }  


    calib_2_pose_13 = {
            "base": math.radians(-73.78),
            "shoulder": math.radians(-134.23),
            "elbow": math.radians(55.01),
            "wrist1": math.radians(-22.27),
            "wrist2": math.radians(-87.82),
            "wrist3": math.radians(-29.12)                   # position ok
        }  


    calib_2_pose_14 = {
            "base": math.radians(-83.36),
            "shoulder": math.radians(-131.19),
            "elbow": math.radians(72.92),
            "wrist1": math.radians(-33.95),
            "wrist2": math.radians(-83.14),
            "wrist3": math.radians(-42.55)                   # position ok
        }  



    calib_2_pose_15 = {
            "base": math.radians(-103.83),
            "shoulder": math.radians(-136.93),
            "elbow": math.radians(72.93),
            "wrist1": math.radians(-35.21),
            "wrist2": math.radians(-83.07),
            "wrist3": math.radians(-59.73)                   # position ok

        }  


    calib_2_pose_16 = {
            "base": math.radians(-64.63),
            "shoulder": math.radians(-84.13),
            "elbow": math.radians(34.81),
            "wrist1": math.radians(-23.85),
            "wrist2": math.radians(-88.88),
            "wrist3": math.radians(-23.44)                   # position ok
        }  


    calib_2_pose_17 = {
            "base": math.radians(-85.24),
            "shoulder": math.radians(-78.43),
            "elbow": math.radians(35.15),
            "wrist1": math.radians(-19.55),
            "wrist2": math.radians(-84.13),
            "wrist3": math.radians(-46.88)                   # position ok
        }  



    calib_2_pose_18 = {
            "base": math.radians(-57.36),
            "shoulder": math.radians(-116.36),
            "elbow": math.radians(61.52),
            "wrist1": math.radians(-36.36),
            "wrist2": math.radians(-85.44),
            "wrist3": math.radians(-16.53)                   # position ok 
        }  



    calib_2_pose_19 = {
            "base": math.radians(-73.12),
            "shoulder": math.radians(-116.92),
            "elbow": math.radians(61.51),
            "wrist1": math.radians(-31.69),
            "wrist2": math.radians(-85.96),
            "wrist3": math.radians(-30.42)                   # position ok
        }  





    calib_flipped_pose_1 = {
            "base": math.radians(-223.65),
            "shoulder": math.radians(-67.52),
            "elbow": math.radians(-55.04),
            "wrist1": math.radians(-144.25),
            "wrist2": math.radians(-267.11),
            "wrist3": math.radians(-16.00)                   # position ok
        }  


    calib_flipped_pose_2 = {
            "base": math.radians(-223.65),
            "shoulder": math.radians(-56.25),
            "elbow": math.radians(-77.84),
            "wrist1": math.radians(-128.02),
            "wrist2": math.radians(-267.22),
            "wrist3": math.radians(-6.20)                   # position ok
        }  


    calib_flipped_pose_3 = {
            "base": math.radians(-223.60),
            "shoulder": math.radians(-72.55),
            "elbow": math.radians(-60.84),
            "wrist1": math.radians(-137.49),
            "wrist2": math.radians(-267.27),
            "wrist3": math.radians(-6.53)                   # position ok
        }  


    calib_flipped_pose_4 = {
            "base": math.radians(-240.30),
            "shoulder": math.radians(-64.95),
            "elbow": math.radians(-61.59),
            "wrist1": math.radians(-138.77),
            "wrist2": math.radians(-265.23),
            "wrist3": math.radians(-30.25)                   # position ok
        }  


    calib_flipped_pose_5 = {
            "base": math.radians(-240.00),
            "shoulder": math.radians(-77.77),
            "elbow": math.radians(-61.45),
            "wrist1": math.radians(-135.09),
            "wrist2": math.radians(-265.32),
            "wrist3": math.radians(-30.22)                   # position ok
        }  


    calib_flipped_pose_6 = {
            "base": math.radians(-234.22),
            "shoulder": math.radians(-77.70),
            "elbow": math.radians(-55.77),
            "wrist1": math.radians(-139.93),
            "wrist2": math.radians(-265.49),
            "wrist3": math.radians(-30.22)                   # position ok
        }  


    calib_flipped_pose_7 = {
            "base": math.radians(-189.01),
            "shoulder": math.radians(-77.77),
            "elbow": math.radians(-55.77),
            "wrist1": math.radians(-139.92),
            "wrist2": math.radians(-265.49),
            "wrist3": math.radians(32.81)                   # position ok
        }  


    calib_flipped_pose_8 = {
            "base": math.radians(-189.00),
            "shoulder": math.radians(-29.83),
            "elbow": math.radians(-80.80),
            "wrist1": math.radians(-136.03),
            "wrist2": math.radians(-265.55),
            "wrist3": math.radians(32.74)                   # position ok
        }  





    calib_flipped_pose_9 = {
            "base": math.radians(-208.14),
            "shoulder": math.radians(-29.17),
            "elbow": math.radians(-85.87),
            "wrist1": math.radians(-132.49),
            "wrist2": math.radians(-265.49),
            "wrist3": math.radians(1.82)                   # position ok
        }  



    calib_3_pose_1 = {
            "base": math.radians(-218.22),
            "shoulder": math.radians(-79.17),
            "elbow": math.radians(-42.94),
            "wrist1": math.radians(-148.17),
            "wrist2": math.radians(-265.51),
            "wrist3": math.radians(-2)                   # position ok
        }  


    calib_3_pose_2 = {
            "base": math.radians(-218.24),
            "shoulder": math.radians(-92.07),
            "elbow": math.radians(-42.51),
            "wrist1": math.radians(-146.74),
            "wrist2": math.radians(-265.55),
            "wrist3": math.radians(-6.02)                   # position ok
        }  


    calib_3_pose_3 = {
            "base": math.radians(-208.12),
            "shoulder": math.radians(-89.12),
            "elbow": math.radians(-28.51),
            "wrist1": math.radians(-157.39),
            "wrist2": math.radians(-263.96),
            "wrist3": math.radians(2.42)                   # position ok
        }  



    calib_3_pose_4 = {
            "base": math.radians(-208.77),
            "shoulder": math.radians(-46.24),
            "elbow": math.radians(-82.51),
            "wrist1": math.radians(-129.69),
            "wrist2": math.radians(-269.06),
            "wrist3": math.radians(1.92)                   # position ok 
        }  


    calib_3_pose_5 = {
            "base": math.radians(-163.82),
            "shoulder": math.radians(-67.11),
            "elbow": math.radians(-61.51),
            "wrist1": math.radians(-143.09),
            "wrist2": math.radians(-268.44),
            "wrist3": math.radians(66.01)                   # position ok 
        }  


    calib_3_pose_6 = {
            "base": math.radians(-58.08),
            "shoulder": math.radians(-115.92),
            "elbow": math.radians(56.51),
            "wrist1": math.radians(132.69),
            "wrist2": math.radians(-268.88),
            "wrist3": math.radians(157.45)                   # position ok (risky!)
        }  



    calib_3_pose_7 = {
            "base": math.radians(-48.99),
            "shoulder": math.radians(-115.66),
            "elbow": math.radians(56.47),
            "wrist1": math.radians(132.69),
            "wrist2": math.radians(-268.88),
            "wrist3": math.radians(174.68)                   # position ok
        }  




    calib_3_pose_8 = {
            "base": math.radians(-48.99),
            "shoulder": math.radians(-121.26),
            "elbow": math.radians(77.23),
            "wrist1": math.radians(117.01),
            "wrist2": math.radians(-268.88),
            "wrist3": math.radians(177.99)                   # position ok
        }  




    calib_3_pose_9 = {
            "base": math.radians(-70.44),
            "shoulder": math.radians(-124.13),
            "elbow": math.radians(89.04),
            "wrist1": math.radians(108.47),
            "wrist2": math.radians(-267.04),
            "wrist3": math.radians(147.77)                   # position ok
        }  



    calib_3_pose_11 = {
            "base": math.radians(-216.77),
            "shoulder": math.radians(-77.57),
            "elbow": math.radians(-43.56),
            "wrist1": math.radians(-149.27),
            "wrist2": math.radians(-261.91),
            "wrist3": math.radians(-2.4)                   # position ok
        }  


    calib_3_pose_12 = {
            "base": math.radians(-218.24),
            "shoulder": math.radians(-91.07),
            "elbow": math.radians(-43.51),
            "wrist1": math.radians(-145.74),
            "wrist2": math.radians(-264.55),
            "wrist3": math.radians(-6.55)                   # position ok
        }  


    calib_3_pose_13 = {
            "base": math.radians(-209.12),
            "shoulder": math.radians(-89.82),
            "elbow": math.radians(-29.11),
            "wrist1": math.radians(-156.79),
            "wrist2": math.radians(-265.16),
            "wrist3": math.radians(2.32)                   # position ok
        }  



    calib_3_pose_14 = {
            "base": math.radians(-208.77),
            "shoulder": math.radians(-47.24),
            "elbow": math.radians(-82.61),
            "wrist1": math.radians(-128.99),
            "wrist2": math.radians(-267.96),
            "wrist3": math.radians(2.72)                   # position ok 
        }  


    calib_3_pose_15 = {
            "base": math.radians(-165.12),
            "shoulder": math.radians(-68.41),
            "elbow": math.radians(-62.41),
            "wrist1": math.radians(-142.69),
            "wrist2": math.radians(-269.14),
            "wrist3": math.radians(68.01)                   # position ok 
        }  


    calib_3_pose_16 = {
            "base": math.radians(-59.08),
            "shoulder": math.radians(-117.92),
            "elbow": math.radians(57.51),
            "wrist1": math.radians(133.69),
            "wrist2": math.radians(-269.18),
            "wrist3": math.radians(159.05)                   # position ok (risky!)
        }  



    calib_3_pose_17 = {
            "base": math.radians(-47.99),
            "shoulder": math.radians(-116.36),
            "elbow": math.radians(58.17),
            "wrist1": math.radians(131.79),
            "wrist2": math.radians(-269.28),
            "wrist3": math.radians(175.68)                   # position ok
        }  




    calib_3_pose_18 = {
            "base": math.radians(-49.99),
            "shoulder": math.radians(-123.16),
            "elbow": math.radians(76.23),
            "wrist1": math.radians(118.01),
            "wrist2": math.radians(-267.98),
            "wrist3": math.radians(179.99)                   # position ok
        }  




    calib_3_pose_19 = {
            "base": math.radians(-71.04),
            "shoulder": math.radians(-123.53),
            "elbow": math.radians(89.54),
            "wrist1": math.radians(107.47),
            "wrist2": math.radians(-266.24),
            "wrist3": math.radians(149.77)                   # position ok
        }  

















    camera_calib_1 = {
            "base": math.radians(-46.28),
            "shoulder": math.radians(-115.46),
            "elbow": math.radians(72.37),
            "wrist1": math.radians(-49.33),
            "wrist2": math.radians(-89.68),
            "wrist3": math.radians(-4.29)                   # position 
        }  
    

    camera_calib_2 = {
            "base": math.radians(-46.00),
            "shoulder": math.radians(-114.74),
            "elbow": math.radians(59.42),
            "wrist1": math.radians(-37.33),
            "wrist2": math.radians(-89.68),
            "wrist3": math.radians(-4.15)                   # position 
        }  
    
    camera_calib_3 = {
            "base": math.radians(-107.55),
            "shoulder": math.radians(-122.64),
            "elbow": math.radians(55.08),
            "wrist1": math.radians(-29.17),
            "wrist2": math.radians(-89.68),
            "wrist3": math.radians(-62.87)                   # position 
        }  
    
    camera_calib_4 = {
            "base": math.radians(-81.55),
            "shoulder": math.radians(-107.74),
            "elbow": math.radians(42.08),
            "wrist1": math.radians(-36.36),
            "wrist2": math.radians(-90.86),
            "wrist3": math.radians(-38.58)                   # position 
        }  
    

    camera_calib_5 = {
            "base": math.radians(-80.97),
            "shoulder": math.radians(-99.94),
            "elbow": math.radians(43.68),
            "wrist1": math.radians(-33.83),
            "wrist2": math.radians(-90.68),
            "wrist3": math.radians(-38.58)                   # position 
        }  

    camera_calib_6 = {
            "base": math.radians(-68.85),
            "shoulder": math.radians(-127.64),
            "elbow": math.radians(81.08),
            "wrist1": math.radians(-52.58),
            "wrist2": math.radians(-93.17),
            "wrist3": math.radians(-25.54)                   # position 
        }  

    camera_calib_7 = {
            "base": math.radians(-45.39),
            "shoulder": math.radians(-155.64),
            "elbow": math.radians(85.40),
            "wrist1": math.radians(-43.46),
            "wrist2": math.radians(-90.87),
            "wrist3": math.radians(-2.93)                   # position 
        }  
    
    camera_calib_8 = {
            "base": math.radians(-45.48),
            "shoulder": math.radians(-76.22),
            "elbow": math.radians(34.79),
            "wrist1": math.radians(-30.17),
            "wrist2": math.radians(-93.22),
            "wrist3": math.radians(-2.93)                   # position 
        }  
    
    camera_calib_9 = {
            "base": math.radians(-47.67),
            "shoulder": math.radians(-92.86),
            "elbow": math.radians(12.95),
            "wrist1": math.radians(-12.52),
            "wrist2": math.radians(-90.85),
            "wrist3": math.radians(-13.64)                   # position 
        }  
    



    camera_calib_poses = [
        camera_calib_1,
        camera_calib_2,
        camera_calib_3,
        camera_calib_4,
        camera_calib_5,
        camera_calib_6,
        camera_calib_7,
        camera_calib_8,
        camera_calib_9,
        ]
    



    new_handeye_pose_1 = {
            "base": math.radians(-52.00),
            "shoulder": math.radians(-150.00),
            "elbow": math.radians(112.50),
            "wrist1": math.radians(-69.00),
            "wrist2": math.radians(-96.00),
            "wrist3": math.radians(-10.00)                   # position 
        } 
    
    new_handeye_pose_2 = {
            "base": math.radians(-52.00),
            "shoulder": math.radians(-139.00),
            "elbow": math.radians(106.00),
            "wrist1": math.radians(-69.33),
            "wrist2": math.radians(-96.00),
            "wrist3": math.radians(-10.00)                   # position 
        }     

    new_handeye_pose_3 = {
            "base": math.radians(-52.00),
            "shoulder": math.radians(-123.50),
            "elbow": math.radians(96.00),
            "wrist1": math.radians(-62.50),
            "wrist2": math.radians(-96.00),
            "wrist3": math.radians(-10.00)                   # position 
        }

    
    new_handeye_pose_4 = {
            "base": math.radians(-52.00),
            "shoulder": math.radians(-109.00),
            "elbow": math.radians(87.00),
            "wrist1": math.radians(-62.00),
            "wrist2": math.radians(-96.00),
            "wrist3": math.radians(-10.00)                   # position 
        }     
    
    new_handeye_pose_5 = {
            "base": math.radians(-52.00),
            "shoulder": math.radians(-103.50),
            "elbow": math.radians(85.50),
            "wrist1": math.radians(-61.00),
            "wrist2": math.radians(-96.00),
            "wrist3": math.radians(-10.00)                   # position 
        }     
    
    new_handeye_pose_6 = {
            "base": math.radians(-51.00),
            "shoulder": math.radians(-90.50),
            "elbow": math.radians(70.70),
            "wrist1": math.radians(-50.50),
            "wrist2": math.radians(-96.00),
            "wrist3": math.radians(-10.00)                   # position 
        } 
        
    new_handeye_pose_7 = {
            "base": math.radians(-51.20),
            "shoulder": math.radians(-82.44),
            "elbow": math.radians(56.90),
            "wrist1": math.radians(-39.73),
            "wrist2": math.radians(-95.44),
            "wrist3": math.radians(-10.00)                   # position 
        } 
        
    new_handeye_pose_8 = {
            "base": math.radians(-50.80),
            "shoulder": math.radians(-75.80),
            "elbow": math.radians(55.00),
            "wrist1": math.radians(-41.00),
            "wrist2": math.radians(-95.50),
            "wrist3": math.radians(-10.00)                   # position 
        } 
        
    new_handeye_pose_9 = {
            "base": math.radians(-30.33),
            "shoulder": math.radians(-85.80),
            "elbow": math.radians(55.00),
            "wrist1": math.radians(-41.00),
            "wrist2": math.radians(-95.50),
            "wrist3": math.radians(13.10)                   # position 
        } 
        
    new_handeye_pose_10 = {
            "base": math.radians(-48.33),
            "shoulder": math.radians(-89.11),
            "elbow": math.radians(55.00),
            "wrist1": math.radians(-41.00),
            "wrist2": math.radians(-96.00),
            "wrist3": math.radians(-8.33)                   # position 
        } 
        
    new_handeye_pose_11 = {
            "base": math.radians(-63.33),
            "shoulder": math.radians(-89.33),
            "elbow": math.radians(51.11),
            "wrist1": math.radians(-36.44),
            "wrist2": math.radians(-95.50),
            "wrist3": math.radians(-20.00)                   # position 
        } 
        
    new_handeye_pose_12 = {
            "base": math.radians(-77.33),
            "shoulder": math.radians(-89.00),
            "elbow": math.radians(50.88),
            "wrist1": math.radians(-39.66),
            "wrist2": math.radians(-95.55),
            "wrist3": math.radians(-38.55)                   # position 
        } 
        
    new_handeye_pose_13 = {
            "base": math.radians(-66.66),
            "shoulder": math.radians(-127.77),
            "elbow": math.radians(86.44),
            "wrist1": math.radians(-61.11),
            "wrist2": math.radians(-95.55),
            "wrist3": math.radians(-25.00)                   # position 
        } 
        
    new_handeye_pose_14 = {
            "base": math.radians(-46.66),
            "shoulder": math.radians(-136.11),
            "elbow": math.radians(93.33),
            "wrist1": math.radians(-60.11),
            "wrist2": math.radians(-95.55),
            "wrist3": math.radians(-3.00)                   # position 
        } 
        
    new_handeye_poses = [
        new_handeye_pose_1,
        new_handeye_pose_2,
        new_handeye_pose_3,
        new_handeye_pose_4,
        new_handeye_pose_5,
        new_handeye_pose_6,
        new_handeye_pose_7,
        new_handeye_pose_8,
        new_handeye_pose_9,
        new_handeye_pose_10,
        new_handeye_pose_11,
        new_handeye_pose_12,
        new_handeye_pose_13,
        new_handeye_pose_14,
    ]

    poses_2 = [
        calib_2_pose_1,
        calib_2_pose_2,
        calib_2_pose_3,
        calib_2_pose_4,
        calib_2_pose_5,
        calib_2_pose_6,
        calib_2_pose_7,
        calib_2_pose_8,
        calib_2_pose_9,
        calib_2_pose_10,
        calib_2_pose_11,
        calib_2_pose_12,
        calib_2_pose_13,
        calib_2_pose_14,
        calib_2_pose_15,
        calib_2_pose_16,
        calib_2_pose_17,
        calib_2_pose_18,
        calib_2_pose_19,
        calib_pose_1,
        calib_pose_2,
        calib_pose_3,
        calib_pose_4,
        calib_pose_5,
        calib_pose_6,
        calib_pose_7,
        calib_pose_8,
        calib_pose_9,
        calib_pose_10,
        # calib_flipped_pose_1,
        # calib_flipped_pose_2,
        # calib_flipped_pose_3,
        # calib_flipped_pose_4,
        # calib_flipped_pose_5,
        # calib_flipped_pose_6,
        # calib_flipped_pose_7,
        # calib_flipped_pose_8,
        # calib_flipped_pose_9,       
        # calib_3_pose_1,
        # calib_3_pose_2,
        # calib_3_pose_3,
        # calib_3_pose_4,
        # calib_3_pose_5,
        # calib_3_pose_6,
        # calib_3_pose_7,
        # calib_3_pose_8,
        # calib_3_pose_9,
        # calib_3_pose_11,
        # calib_3_pose_12,
        # calib_3_pose_13,
        # calib_3_pose_14,
        # calib_3_pose_15,
        # calib_3_pose_16,
        # calib_3_pose_17,
        # calib_3_pose_18,
        # calib_3_pose_19,
        # home,
        new_handeye_pose_1,
        new_handeye_pose_2,
        new_handeye_pose_3,
        new_handeye_pose_4,
        new_handeye_pose_5,
        new_handeye_pose_6,
        new_handeye_pose_7,
        new_handeye_pose_8,
        new_handeye_pose_9,
        new_handeye_pose_10,
        new_handeye_pose_11,
        new_handeye_pose_12,
        new_handeye_pose_13,
        new_handeye_pose_14,
        ]