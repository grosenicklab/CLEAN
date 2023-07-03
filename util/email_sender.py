# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.
                                  
import smtplib

def send_email( recipient, subject, msg, sender, password ):
    """                                             
    Send and email from the enlightenment.fail account.                                       
    """
    body = "" + msg + ""

    headers = ["From: " + sender,
               "Subject: " + subject,
               "To: " + recipient,
               "MIME-Version: 1.0",
               "Content-Type: text/html"]
    headers = "\r\n".join(headers)

    session = smtplib.SMTP('smtp.gmail.com', 587)

    session.ehlo()
    session.starttls()
    session.ehlo
    session.login(sender, password)

    session.sendmail(sender, recipient, headers + "\r\n\r\n" + body)
    session.quit()

if __name__ == "__main__":
    pass
