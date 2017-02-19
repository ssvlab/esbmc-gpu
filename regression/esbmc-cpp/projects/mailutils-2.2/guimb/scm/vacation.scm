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

;;;; This module provides "vacation" extension

;;; vacation example:
;;; vacation :days 18
;;;          :aliases ["gray@gnu.org", "gray@mirddin.farlep.net"]
;;;          :addresses ["bug-mailutils@gnu.org","bug-inetutils@gnu.org"]
;;;          :subject "I'm on vacation"
;;;          :mime
;;;          text:
;;;   I am on vacation until July 22. I'll attend your message as soon
;;;   as I'm back.
;;; .
;;;
;;; Additionally, the :sender flag may be used to debug the script.

;; Debugging flag
(define vacation-debug #f)

;; Each entry is (cons SENDER DATE), where SENDER is the sender email
;; address (lowercase) and DATE is the date where the first message
;; from this sender was received.
(define vacation-db '())

(define (vacation-downcase name)
  (let ((len (string-length name)))
    (do ((i 0 (1+ i)))
	((= i len) name)
      (string-set! name i (char-downcase (string-ref name i))))))

(define (vacation-db-name)
  (let ((pwd (mu-getpwuid (getuid))))
    (string-append (vector-ref pwd 5) "/.vacation.db")))

(define (vacation-db-load)
  (catch #t
	 (lambda ()
	   (call-with-input-file (vacation-db-name)
	     (lambda (port)
	       (set! vacation-db (read port)))))
	 (lambda args args)))

(define (vacation-db-save)
  (catch #t
	 (lambda ()
	   (let ((mask (umask #o077)))
	     (call-with-output-file (vacation-db-name)
	       (lambda (port)
		 (display ";; Vacation database file\n" port)
		 (display ";; Generated automatically. Please do not edit\n"
			  port)
		 (write vacation-db port)))
	     (umask mask)))
	 (lambda args args)))

(define (vacation-db-lookup sender days)
  (vacation-db-load)  
  (let ((val (assoc (vacation-downcase sender) vacation-db)))
    (cond
     (val
      (cond
       ((and days (> days 0))
	(<= (- (car (gettimeofday)) (cdr val)) (* days 86400)))
       (else
	#t)))
     (else
      #f))))

(define (vacation-db-update msg)
  (let* ((sender (vacation-downcase (mu-message-get-sender msg)))
	 (date (car (gettimeofday)))
	 (val (assoc sender vacation-db)))
    (cond
     (val
      (set-cdr! val date))
     (else
      (set! vacation-db (append vacation-db (list
					     (cons sender date)))))))
  (vacation-db-save))

(define vacation-noreply-senders
  (list
   ".*-REQUEST@.*"
   ".*-RELAY@.*"
   ".*-OWNER@.*"
   "OWNER-.*"
   "postmaster@.*"
   "UUCP@.*"
   "MAILER@.*"
   "MAILER-DAEMON@.*"))

(define (vacation-reply? msg aliases addresses days)
  (let ((sender (mu-message-get-sender msg)))
    (and
     ;; No message will be sent unless an alias is part of either
     ;; the "To:" or "Cc:" headers of the mail.
     (call-with-current-continuation
      (lambda (exit)
	(for-each
	 (lambda (hdr)
	   (cond
	    (hdr
	     (let ((count (mu-address-get-count hdr)))
	       (do ((i 1 (1+ i)))
		   ((> i count) #f)
		 (let ((email (mu-address-get-email hdr i)))
		   (for-each
		    (lambda (alias)
		      (if (string-ci=? alias email)
			  (exit #t)))
		    aliases)))))))
	 (list (mu-message-get-header msg "To")
	       (mu-message-get-header msg "Cc")))
	#f))

     ;; Messages sent from one of the vacation-noreply-senders are not
     ;; responded to
     (call-with-current-continuation
      (lambda (exit)
	(do ((explist (append vacation-noreply-senders addresses)
		      (cdr explist)))
	    ((null? explist) #t)
	  (let ((rx (make-regexp (car explist) regexp/icase)))
	    (if (regexp-exec rx sender)
		(exit #f))))))

     ;; Messages with Precedence: bulk or junk are not responded to
     (let ((prec (mu-message-get-header msg "Precedence")))
       (not (and prec (or (string-ci=? prec "bulk")
			  (string-ci=? prec "junk")))))

     ;; Senders already in the database get no response
     (not (vacation-db-lookup sender days)))))

(define (vacation-send-reply subject text sender)
  (let ((sender "root@localhost")
	(mesg (mu-message-create)))
    (let ((port (mu-message-get-port mesg "w")))
      (display text port)
      (close-output-port port))
    (mu-message-set-header mesg "X-Sender"
			   (string-append "vacation.scm, " mu-package-string)
			   #t)
    (mu-message-send mesg #f #f sender)))

(define (action-vacation text . opt)
  (sieve-verbose-print "VACATION")
  (set! vacation-debug (member #:debug opt))
  (if vacation-debug
      (begin
	(display sieve-current-message)(display ": ")))
  (cond
   ((vacation-reply? sieve-current-message
		     (append (list sieve-my-email)
			     (sieve-get-opt-arg opt #:aliases '()))
		     (sieve-get-opt-arg opt #:addresses '())
		     (sieve-get-opt-arg opt #:days #f))
    (vacation-send-reply (sieve-get-opt-arg
			  opt #:subject
			  (string-append "Re: "
					 (mu-message-get-header
					  sieve-current-message
					  "Subject")))
			 text
			 (sieve-get-opt-arg
			  opt #:sender
			  (mu-message-get-sender sieve-current-message)))
    (vacation-db-update sieve-current-message)
    (if vacation-debug
	(display "WILL REPLY\n")))
   (vacation-debug
    (display "WILL NOT REPLY\n"))))

;;; Register action
(if sieve-parser
    (sieve-register-action "vacation"
			   action-vacation
			   (list 'string)
			   (list (cons "days" 'number)
				 (cons "addresses" 'string-list)
				 (cons "aliases" 'string-list)
				 (cons "subject" 'string)
				 (cons "sender" 'string)
				 (cons "mime" #f)
				 (cons "debug" #f))))

