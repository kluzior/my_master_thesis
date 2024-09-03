class CmdGenerator:

    def basic(message):
        return message.encode('utf-8')
    
    def pose_convert_to_tcp_frame(pose):                # FIX THIS NAME!!! pose -> joints
        cmd = f"({pose.get('base')}," +\
                f"{pose.get('shoulder')}," +\
                f"{pose.get('elbow')}," +\
                f"{pose.get('wrist1')}," +\
                f"{pose.get('wrist2')}," +\
                f"{pose.get('wrist3')}" +\
                f")" + "\n"
        return cmd.encode('utf-8')

    def joints_convert_to_tcp_frame(pose):                # pose_convert_to_tcp_frame to this function
        cmd = f"({pose.get('base')}," +\
                f"{pose.get('shoulder')}," +\
                f"{pose.get('elbow')}," +\
                f"{pose.get('wrist1')}," +\
                f"{pose.get('wrist2')}," +\
                f"{pose.get('wrist3')}" +\
                f")" + "\n"
        return cmd.encode('utf-8')

    def pose_convert_to_tcp_frame2(pose):
        cmd = f"({pose[0]}, {pose[1]}, {pose[2]}, {pose[3]}, {pose[4]}, {pose[5]})" + "\n"
        return cmd.encode('utf-8')

    def pose_convert_to_tcp_frame3(pose): 
        cmd = f"({pose.get('x')}," +\
                f"{pose.get('y')}," +\
                f"{pose.get('z')}," +\
                f"{pose.get('Rx')}," +\
                f"{pose.get('Ry')}," +\
                f"{pose.get('Rz')}" +\
                f")" + "\n"
        return cmd.encode('utf-8')
    

    def movej_gen(pose, acc=0.5, vel=0.5):
        cmd = f"movej([{pose.get('base')}," +\
                f"{pose.get('shoulder')}," +\
                f"{pose.get('elbow')}," +\
                f"{pose.get('wrist1')}," +\
                f"{pose.get('wrist2')}," +\
                f"{pose.get('wrist3')}" +\
                f"], a = {acc}, v = {vel})" + "\n"
        return cmd.encode('utf-8')
    
    def set_output(out_no):
        cmd = f"set_digital_out({out_no}, True)" + "\n"
        return cmd.encode('utf-8')

    def reset_output(out_no):
        cmd = f"set_digital_out({out_no}, False)" + "\n"
        return cmd.encode('utf-8')