;;;; GNU Mailutils -- a suite of utilities for electronic mail
;;;; Copyright (C) 1999, 2000, 2001, 2006, 2007, 2010 Free Software
;;;; Foundation, Inc.
;;;;
;;;; GNU Mailutils is free software; you can redistribute it and/or modify
;;;; it under the terms of the GNU General Public License as published by
;;;; the Free Software Foundation; either version 3, or (at your option)
;;;; any later version.
;;;; 
;;;; GNU Mailutils is distributed in the hope that it will be useful,
;;;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;;;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;;;; GNU General Public License for more details.
;;;; 
;;;; You should have received a copy of the GNU General Public License along
;;;; with GNU Mailutils; if not, write to the Free Software Foundation,
;;;; Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

;;;; This module provides sieve's "reject" action.

(define sieve-option-quote #t)

(define (action-reject reason)
  (sieve-verbose-print "REJECT")
  (handle-exception
   (let ((mime (mu-mime-create 0))
	 (datestr (strftime "%a, %b %d %H:%M:%S %Y %Z"
			    (localtime (current-time))))
	 (sender (mu-message-get-sender sieve-current-message)))
     (let* ((mesg (mu-message-create))
	    (port (mu-message-get-port mesg "w")))
       
       (display "The original message was received at " port)
       (display datestr port)
       (newline port)
       (display "from " port)
       (display sender port)
       (display ".\n" port)
       
       (display "Message was refused by recipient's mail filtering program.\n"
		port)
       (display "Reason given was as follows:\n" port)
       (newline port)
       (display reason port)

       (close-output-port port)
       (mu-mime-add-part mime mesg))

     ;; message/delivery-status
     (let* ((mesg (mu-message-create))
	    (port (mu-message-get-port mesg "w")))
       (mu-message-set-header mesg "Content-Type" "message/delivery-status")
       
       (display (string-append "Reporting-UA: sieve; GNU "
			       mu-package-string "\n") port)
       (display (string-append "Arrival-Date: " datestr "\n") port)
       (newline port)
       
       (display (string-append "Final-Recipient: RFC822; " sieve-my-email "\n")
		port)
       
       (display "Action: deleted\n" port);
       (display "Disposition: automatic-action/MDN-sent-automatically;deleted\n"
		port)
       (display (string-append
		 "Last-Attempt-Date: " datestr "\n") port)
       
       (close-output-port port)
       (mu-mime-add-part mime mesg))
     
     ;; Quote original message
     (let* ((mesg (mu-message-create))
	    (port (mu-message-get-port mesg "w"))
	    (in-port (mu-message-get-port sieve-current-message "r" #t)))
       (mu-message-set-header mesg "Content-Type" "message/rfc822")
       
       (do ((line (read-line in-port) (read-line in-port)))
	   ((eof-object? line) #f)
	 (display line port)
	 (newline port))
       
       (close-input-port in-port)
       (close-output-port port)
       (mu-mime-add-part mime mesg))
     
     (mu-message-send (mu-mime-get-message mime) #f sieve-daemon-email sender)
     (mu-message-delete sieve-current-message))))

;;; Register action
(if sieve-parser
    (sieve-register-action "reject" action-reject (list 'string) '()))
      
      

      
    
