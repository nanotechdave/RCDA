import logging

def main():
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(levelname)s %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        filename = "basic.log",
    )
    logging.debug("this is a debug message")
    logging.info("now we are printing info")
    logging.warning("this is a warning")
    logging.error("this is an error")
    logging.critical("critical message")

    
    return



if __name__ == "__main__":
    main()
    
