"""
Email Alert Module - Fall Detection System
Sends email alerts with fall detection screenshots
when a fall is detected from webcam or uploaded video.

Supports any SMTP provider:
  - Gmail:   smtp.gmail.com  port 587
  - Outlook: smtp.office365.com port 587
  - Yahoo:   smtp.mail.yahoo.com port 587

NOTE for Gmail users:
  - Enable 2FA on your Google account
  - Generate an App Password at: myaccount.google.com/apppasswords
  - Use that App Password here (NOT your regular Gmail password)
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path
import traceback
import os


def send_fall_alert(
    recipient_email: str,
    sender_email: str,
    sender_password: str,
    image_path: str = None,
    detection_info: dict = None,
    source: str = 'video',
    smtp_host: str = 'smtp.gmail.com',
    smtp_port: int = 587,
) -> tuple:
    """
    Send a fall detection email alert with an optional screenshot attached.

    Args:
        recipient_email  : email address(es) to notify - comma-separated string or list
        sender_email     : SMTP sender address (your email)
        sender_password  : SMTP password / App Password
        image_path       : path to the screenshot to attach (optional)
        detection_info   : dict with keys: timestamp, confidence, source_file, etc.
        source           : 'video' or 'webcam'
        smtp_host        : SMTP server hostname
        smtp_port        : SMTP server port (587 for TLS)

    Returns:
        (success: bool, message: str)
    """
    if not recipient_email or not sender_email or not sender_password:
        return False, "Email settings incomplete (missing recipient, sender, or password)"

    try:
        # Support comma-separated recipients
        if isinstance(recipient_email, str):
            recipients = [r.strip() for r in recipient_email.split(',') if r.strip()]
        else:
            recipients = list(recipient_email)

        # Build info dict for display
        info = detection_info or {}
        timestamp     = info.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        confidence    = info.get('confidence', 0)
        conf_pct      = f"{float(confidence)*100:.0f}%" if confidence else "N/A"
        source_file   = info.get('source_file', 'Unknown')
        detection_num = info.get('detection_num', 1)

        source_label = 'Webcam (Live)' if source == 'webcam' else f'Video Upload ({source_file})'

        # ── Email subject ──────────────────────────────────────────────────────
        subject = f"🚨 FALL DETECTED - {datetime.now().strftime('%d %b %Y %H:%M:%S')}"

        # ── HTML body ──────────────────────────────────────────────────────────
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5;">

        <div style="max-width: 600px; margin: 30px auto; background: white;
                    border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden;">

            <!-- Header -->
            <div style="background: linear-gradient(135deg, #dc3545, #c82333);
                        padding: 30px; text-align: center;">
                <h1 style="color: white; margin: 0; font-size: 28px;">
                    🚨 FALL DETECTED
                </h1>
                <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0; font-size: 16px;">
                    Immediate attention may be required
                </p>
            </div>

            <!-- Body -->
            <div style="padding: 30px;">

                <!-- Alert Box -->
                <div style="background: #fff3cd; border: 2px solid #ffc107;
                            border-radius: 8px; padding: 20px; margin-bottom: 25px;">
                    <h2 style="color: #856404; margin: 0 0 10px;">
                        ⚠️ Alert &nbsp;#{detection_num}
                    </h2>
                    <p style="color: #856404; margin: 0; font-size: 15px;">
                        A potential fall event has been automatically detected.
                        Please check on the individual immediately.
                    </p>
                </div>

                <!-- Details Table -->
                <h3 style="color: #333; border-bottom: 2px solid #dc3545;
                           padding-bottom: 8px; margin-bottom: 15px;">
                    Detection Details
                </h3>

                <table style="width: 100%; border-collapse: collapse; margin-bottom: 25px;">
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 10px 15px; font-weight: bold;
                                   color: #555; width: 40%; border-radius: 4px;">
                            📅 Date &amp; Time
                        </td>
                        <td style="padding: 10px 15px; color: #333;">
                            {timestamp}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 15px; font-weight: bold; color: #555;">
                            🎯 Confidence
                        </td>
                        <td style="padding: 10px 15px; color: #333;">
                            <span style="background: #dc3545; color: white;
                                         padding: 3px 10px; border-radius: 20px;
                                         font-weight: bold;">
                                {conf_pct}
                            </span>
                        </td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 10px 15px; font-weight: bold; color: #555;">
                            📷 Source
                        </td>
                        <td style="padding: 10px 15px; color: #333;">
                            {source_label}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 15px; font-weight: bold; color: #555;">
                            🔢 Detection #
                        </td>
                        <td style="padding: 10px 15px; color: #333;">
                            {detection_num}
                        </td>
                    </tr>
                </table>

                <!-- Screenshot note -->
                {"<p style='color:#555; font-size:14px;'>📎 <strong>Screenshot attached</strong> — see image below or the attached file.</p>" if image_path else "<p style='color:#aaa; font-size:13px;'>No screenshot available for this detection.</p>"}

                <!-- Footer -->
                <div style="border-top: 1px solid #eee; padding-top: 20px; margin-top: 10px;">
                    <p style="color: #888; font-size: 12px; margin: 0; text-align: center;">
                        Sent automatically by
                        <strong>Fall Detection System</strong> &bull;
                        {datetime.now().strftime('%Y')}
                    </p>
                </div>
            </div>
        </div>

        </body>
        </html>
        """

        # Plain-text fallback
        plain_body = (
            f"FALL DETECTED\n"
            f"{'='*40}\n"
            f"Time:       {timestamp}\n"
            f"Confidence: {conf_pct}\n"
            f"Source:     {source_label}\n"
            f"Detection:  #{detection_num}\n"
            f"{'='*40}\n"
            f"Please check on the person immediately.\n"
        )

        # ── Assemble message ───────────────────────────────────────────────────
        msg = MIMEMultipart('related')
        msg['Subject'] = subject
        msg['From']    = sender_email
        msg['To']      = ', '.join(recipients)

        # Attach alternative (plain + html)
        alt_part = MIMEMultipart('alternative')
        alt_part.attach(MIMEText(plain_body, 'plain'))
        alt_part.attach(MIMEText(html_body,  'html'))
        msg.attach(alt_part)

        # Attach screenshot if available
        if image_path and os.path.exists(str(image_path)):
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()

            # Embed inline in HTML
            img_inline = MIMEImage(img_data)
            img_inline.add_header('Content-ID',          '<fall_screenshot>')
            img_inline.add_header('Content-Disposition', 'inline',
                                  filename=os.path.basename(str(image_path)))
            msg.attach(img_inline)

            # Also attach as downloadable
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(img_data)
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition', 'attachment',
                filename=f"fall_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            msg.attach(attachment)

        # ── Send via SMTP ──────────────────────────────────────────────────────
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())

        recipient_str = ', '.join(recipients)
        print(f"✅ Fall alert email sent to: {recipient_str}")
        return True, f"Email sent to {recipient_str}"

    except smtplib.SMTPAuthenticationError:
        msg = (
            "Authentication failed. "
            "For Gmail, use an App Password (not your regular password). "
            "Enable 2FA at myaccount.google.com, then create an App Password."
        )
        print(f"❌ Email auth error: {msg}")
        return False, msg

    except smtplib.SMTPException as e:
        msg = f"SMTP error: {str(e)}"
        print(f"❌ {msg}")
        return False, msg

    except Exception as e:
        msg = f"Failed to send email: {str(e)}"
        print(f"❌ {msg}")
        traceback.print_exc()
        return False, msg


def send_test_email(
    recipient_email: str,
    sender_email: str,
    sender_password: str,
    smtp_host: str = 'smtp.gmail.com',
    smtp_port: int = 587,
) -> tuple:
    """
    Send a test email to verify SMTP settings are correct.

    Returns:
        (success: bool, message: str)
    """
    test_info = {
        'timestamp':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'confidence':   0.95,
        'source_file':  'test_email',
        'detection_num': 1,
    }
    return send_fall_alert(
        recipient_email  = recipient_email,
        sender_email     = sender_email,
        sender_password  = sender_password,
        image_path       = None,
        detection_info   = {**test_info,
                            'timestamp': f"TEST - {test_info['timestamp']}"},
        source           = 'webcam',
        smtp_host        = smtp_host,
        smtp_port        = smtp_port,
    )
