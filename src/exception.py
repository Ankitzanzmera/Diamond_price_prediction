import sys

def error_message_detail(error_msg,error_detail):
    _,_,exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename

    error_msg = f'Error occured in python script name [{filename}],line number [{exc_tb.tb_lineno}] error message[{str(error_msg)}]'

    return error_msg

class CustomException(Exception):
    def __init__(self,error_msg , error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_message_detail(error_msg,error_detail)

    def __str__(self) -> str:
        return self.error_msg


if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        raise CustomException(e,sys)