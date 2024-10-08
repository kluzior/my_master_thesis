class CmdGenerator:
    def basic(message):
        return message.encode('utf-8')
    
    def joints_convert_to_tcp_frame(pose):
        cmd = f"({pose.get('base')}," +\
                f"{pose.get('shoulder')}," +\
                f"{pose.get('elbow')}," +\
                f"{pose.get('wrist1')}," +\
                f"{pose.get('wrist2')}," +\
                f"{pose.get('wrist3')}" +\
                f")" + "\n"
        return cmd.encode('utf-8')

    def pose_convert_to_tcp_frame(pose): 
        cmd = f"({pose.get('x')}," +\
                f"{pose.get('y')}," +\
                f"{pose.get('z')}," +\
                f"{pose.get('Rx')}," +\
                f"{pose.get('Ry')}," +\
                f"{pose.get('Rz')}" +\
                f")" + "\n"
        return cmd.encode('utf-8')