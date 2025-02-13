from generator.chat_history_control import ControlChatHistoryData
import toml

class IDControl():
    config = toml.load("config/parameter.toml")["help"]["help"]
    


    chat_history_control = ControlChatHistoryData()
    #帮助系统:
    def help(self,order):
        if order == "help":

    #用户会话或消息列出系统
    def list_id(self,order, function):
        if order == "ls user":
            return self.chat_history_control.list_user_ids()


    #用户或会话加载系统
    def load_id(order, function):


