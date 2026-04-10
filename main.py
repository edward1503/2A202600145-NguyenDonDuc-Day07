import sys
from streamlit.web import cli as stcli

if __name__ == '__main__':
    # Thống nhất ứng dụng (không chia backend/frontend) 
    # File main.py trở thành entrypoint khởi chạy nguyên khối qua Streamlit.
    sys.argv = ["streamlit", "run", "streamlit_app.py"]
    sys.exit(stcli.main())
