# 1 "exStbDemo.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "exStbDemo.c"
# 107 "exStbDemo.c"
# 1 "/usr/include/stdio.h" 1 3
# 28 "/usr/include/stdio.h" 3
# 1 "/usr/include/features.h" 1 3
# 335 "/usr/include/features.h" 3
# 1 "/usr/include/sys/cdefs.h" 1 3
# 360 "/usr/include/sys/cdefs.h" 3
# 1 "/usr/include/bits/wordsize.h" 1 3
# 361 "/usr/include/sys/cdefs.h" 2 3
# 336 "/usr/include/features.h" 2 3
# 359 "/usr/include/features.h" 3
# 1 "/usr/include/gnu/stubs.h" 1 3



# 1 "/usr/include/bits/wordsize.h" 1 3
# 5 "/usr/include/gnu/stubs.h" 2 3


# 1 "/usr/include/gnu/stubs-32.h" 1 3
# 8 "/usr/include/gnu/stubs.h" 2 3
# 360 "/usr/include/features.h" 2 3
# 29 "/usr/include/stdio.h" 2 3





# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 214 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 3 4
typedef unsigned int size_t;
# 35 "/usr/include/stdio.h" 2 3

# 1 "/usr/include/bits/types.h" 1 3
# 28 "/usr/include/bits/types.h" 3
# 1 "/usr/include/bits/wordsize.h" 1 3
# 29 "/usr/include/bits/types.h" 2 3


typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;


typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;




__extension__ typedef signed long long int __int64_t;
__extension__ typedef unsigned long long int __uint64_t;







__extension__ typedef long long int __quad_t;
__extension__ typedef unsigned long long int __u_quad_t;
# 131 "/usr/include/bits/types.h" 3
# 1 "/usr/include/bits/typesizes.h" 1 3
# 132 "/usr/include/bits/types.h" 2 3


__extension__ typedef __u_quad_t __dev_t;
__extension__ typedef unsigned int __uid_t;
__extension__ typedef unsigned int __gid_t;
__extension__ typedef unsigned long int __ino_t;
__extension__ typedef __u_quad_t __ino64_t;
__extension__ typedef unsigned int __mode_t;
__extension__ typedef unsigned int __nlink_t;
__extension__ typedef long int __off_t;
__extension__ typedef __quad_t __off64_t;
__extension__ typedef int __pid_t;
__extension__ typedef struct { int __val[2]; } __fsid_t;
__extension__ typedef long int __clock_t;
__extension__ typedef unsigned long int __rlim_t;
__extension__ typedef __u_quad_t __rlim64_t;
__extension__ typedef unsigned int __id_t;
__extension__ typedef long int __time_t;
__extension__ typedef unsigned int __useconds_t;
__extension__ typedef long int __suseconds_t;

__extension__ typedef int __daddr_t;
__extension__ typedef long int __swblk_t;
__extension__ typedef int __key_t;


__extension__ typedef int __clockid_t;


__extension__ typedef void * __timer_t;


__extension__ typedef long int __blksize_t;




__extension__ typedef long int __blkcnt_t;
__extension__ typedef __quad_t __blkcnt64_t;


__extension__ typedef unsigned long int __fsblkcnt_t;
__extension__ typedef __u_quad_t __fsblkcnt64_t;


__extension__ typedef unsigned long int __fsfilcnt_t;
__extension__ typedef __u_quad_t __fsfilcnt64_t;

__extension__ typedef int __ssize_t;



typedef __off64_t __loff_t;
typedef __quad_t *__qaddr_t;
typedef char *__caddr_t;


__extension__ typedef int __intptr_t;


__extension__ typedef unsigned int __socklen_t;
# 37 "/usr/include/stdio.h" 2 3
# 45 "/usr/include/stdio.h" 3
struct _IO_FILE;



typedef struct _IO_FILE FILE;





# 65 "/usr/include/stdio.h" 3
typedef struct _IO_FILE __FILE;
# 75 "/usr/include/stdio.h" 3
# 1 "/usr/include/libio.h" 1 3
# 32 "/usr/include/libio.h" 3
# 1 "/usr/include/_G_config.h" 1 3
# 15 "/usr/include/_G_config.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 16 "/usr/include/_G_config.h" 2 3




# 1 "/usr/include/wchar.h" 1 3
# 78 "/usr/include/wchar.h" 3
typedef struct
{
  int __count;
  union
  {

    unsigned int __wch;



    char __wchb[4];
  } __value;
} __mbstate_t;
# 21 "/usr/include/_G_config.h" 2 3

typedef struct
{
  __off_t __pos;
  __mbstate_t __state;
} _G_fpos_t;
typedef struct
{
  __off64_t __pos;
  __mbstate_t __state;
} _G_fpos64_t;
# 53 "/usr/include/_G_config.h" 3
typedef int _G_int16_t __attribute__ ((__mode__ (__HI__)));
typedef int _G_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int _G_uint16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int _G_uint32_t __attribute__ ((__mode__ (__SI__)));
# 33 "/usr/include/libio.h" 2 3
# 53 "/usr/include/libio.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stdarg.h" 1 3 4
# 43 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stdarg.h" 3 4
typedef __builtin_va_list __gnuc_va_list;
# 54 "/usr/include/libio.h" 2 3
# 170 "/usr/include/libio.h" 3
struct _IO_jump_t; struct _IO_FILE;
# 180 "/usr/include/libio.h" 3
typedef void _IO_lock_t;





struct _IO_marker {
  struct _IO_marker *_next;
  struct _IO_FILE *_sbuf;



  int _pos;
# 203 "/usr/include/libio.h" 3
};


enum __codecvt_result
{
  __codecvt_ok,
  __codecvt_partial,
  __codecvt_error,
  __codecvt_noconv
};
# 271 "/usr/include/libio.h" 3
struct _IO_FILE {
  int _flags;




  char* _IO_read_ptr;
  char* _IO_read_end;
  char* _IO_read_base;
  char* _IO_write_base;
  char* _IO_write_ptr;
  char* _IO_write_end;
  char* _IO_buf_base;
  char* _IO_buf_end;

  char *_IO_save_base;
  char *_IO_backup_base;
  char *_IO_save_end;

  struct _IO_marker *_markers;

  struct _IO_FILE *_chain;

  int _fileno;



  int _flags2;

  __off_t _old_offset;



  unsigned short _cur_column;
  signed char _vtable_offset;
  char _shortbuf[1];



  _IO_lock_t *_lock;
# 319 "/usr/include/libio.h" 3
  __off64_t _offset;
# 328 "/usr/include/libio.h" 3
  void *__pad1;
  void *__pad2;
  void *__pad3;
  void *__pad4;
  size_t __pad5;

  int _mode;

  char _unused2[15 * sizeof (int) - 4 * sizeof (void *) - sizeof (size_t)];

};


typedef struct _IO_FILE _IO_FILE;


struct _IO_FILE_plus;

extern struct _IO_FILE_plus _IO_2_1_stdin_;
extern struct _IO_FILE_plus _IO_2_1_stdout_;
extern struct _IO_FILE_plus _IO_2_1_stderr_;
# 364 "/usr/include/libio.h" 3
typedef __ssize_t __io_read_fn (void *__cookie, char *__buf, size_t __nbytes);







typedef __ssize_t __io_write_fn (void *__cookie, __const char *__buf,
     size_t __n);







typedef int __io_seek_fn (void *__cookie, __off64_t *__pos, int __w);


typedef int __io_close_fn (void *__cookie);
# 416 "/usr/include/libio.h" 3
extern int __underflow (_IO_FILE *);
extern int __uflow (_IO_FILE *);
extern int __overflow (_IO_FILE *, int);
# 458 "/usr/include/libio.h" 3
extern int _IO_getc (_IO_FILE *__fp);
extern int _IO_putc (int __c, _IO_FILE *__fp);
extern int _IO_feof (_IO_FILE *__fp) __attribute__ ((__nothrow__));
extern int _IO_ferror (_IO_FILE *__fp) __attribute__ ((__nothrow__));

extern int _IO_peekc_locked (_IO_FILE *__fp);





extern void _IO_flockfile (_IO_FILE *) __attribute__ ((__nothrow__));
extern void _IO_funlockfile (_IO_FILE *) __attribute__ ((__nothrow__));
extern int _IO_ftrylockfile (_IO_FILE *) __attribute__ ((__nothrow__));
# 488 "/usr/include/libio.h" 3
extern int _IO_vfscanf (_IO_FILE * __restrict, const char * __restrict,
   __gnuc_va_list, int *__restrict);
extern int _IO_vfprintf (_IO_FILE *__restrict, const char *__restrict,
    __gnuc_va_list);
extern __ssize_t _IO_padn (_IO_FILE *, int, __ssize_t);
extern size_t _IO_sgetn (_IO_FILE *, void *, size_t);

extern __off64_t _IO_seekoff (_IO_FILE *, __off64_t, int, int);
extern __off64_t _IO_seekpos (_IO_FILE *, __off64_t, int);

extern void _IO_free_backup_area (_IO_FILE *) __attribute__ ((__nothrow__));
# 76 "/usr/include/stdio.h" 2 3
# 89 "/usr/include/stdio.h" 3


typedef _G_fpos_t fpos_t;




# 141 "/usr/include/stdio.h" 3
# 1 "/usr/include/bits/stdio_lim.h" 1 3
# 142 "/usr/include/stdio.h" 2 3



extern struct _IO_FILE *stdin;
extern struct _IO_FILE *stdout;
extern struct _IO_FILE *stderr;









extern int remove (__const char *__filename) __attribute__ ((__nothrow__));

extern int rename (__const char *__old, __const char *__new) __attribute__ ((__nothrow__));














extern FILE *tmpfile (void) ;
# 188 "/usr/include/stdio.h" 3
extern char *tmpnam (char *__s) __attribute__ ((__nothrow__)) ;





extern char *tmpnam_r (char *__s) __attribute__ ((__nothrow__)) ;
# 206 "/usr/include/stdio.h" 3
extern char *tempnam (__const char *__dir, __const char *__pfx)
     __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;








extern int fclose (FILE *__stream);




extern int fflush (FILE *__stream);

# 231 "/usr/include/stdio.h" 3
extern int fflush_unlocked (FILE *__stream);
# 245 "/usr/include/stdio.h" 3






extern FILE *fopen (__const char *__restrict __filename,
      __const char *__restrict __modes) ;




extern FILE *freopen (__const char *__restrict __filename,
        __const char *__restrict __modes,
        FILE *__restrict __stream) ;
# 274 "/usr/include/stdio.h" 3

# 285 "/usr/include/stdio.h" 3
extern FILE *fdopen (int __fd, __const char *__modes) __attribute__ ((__nothrow__)) ;
# 306 "/usr/include/stdio.h" 3



extern void setbuf (FILE *__restrict __stream, char *__restrict __buf) __attribute__ ((__nothrow__));



extern int setvbuf (FILE *__restrict __stream, char *__restrict __buf,
      int __modes, size_t __n) __attribute__ ((__nothrow__));





extern void setbuffer (FILE *__restrict __stream, char *__restrict __buf,
         size_t __size) __attribute__ ((__nothrow__));


extern void setlinebuf (FILE *__stream) __attribute__ ((__nothrow__));








extern int fprintf (FILE *__restrict __stream,
      __const char *__restrict __format, ...);




extern int printf (__const char *__restrict __format, ...);

extern int sprintf (char *__restrict __s,
      __const char *__restrict __format, ...) __attribute__ ((__nothrow__));





extern int vfprintf (FILE *__restrict __s, __const char *__restrict __format,
       __gnuc_va_list __arg);




extern int vprintf (__const char *__restrict __format, __gnuc_va_list __arg);

extern int vsprintf (char *__restrict __s, __const char *__restrict __format,
       __gnuc_va_list __arg) __attribute__ ((__nothrow__));





extern int snprintf (char *__restrict __s, size_t __maxlen,
       __const char *__restrict __format, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 4)));

extern int vsnprintf (char *__restrict __s, size_t __maxlen,
        __const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 0)));

# 400 "/usr/include/stdio.h" 3





extern int fscanf (FILE *__restrict __stream,
     __const char *__restrict __format, ...) ;




extern int scanf (__const char *__restrict __format, ...) ;

extern int sscanf (__const char *__restrict __s,
     __const char *__restrict __format, ...) __attribute__ ((__nothrow__));
# 443 "/usr/include/stdio.h" 3

# 506 "/usr/include/stdio.h" 3





extern int fgetc (FILE *__stream);
extern int getc (FILE *__stream);





extern int getchar (void);

# 530 "/usr/include/stdio.h" 3
extern int getc_unlocked (FILE *__stream);
extern int getchar_unlocked (void);
# 541 "/usr/include/stdio.h" 3
extern int fgetc_unlocked (FILE *__stream);











extern int fputc (int __c, FILE *__stream);
extern int putc (int __c, FILE *__stream);





extern int putchar (int __c);

# 574 "/usr/include/stdio.h" 3
extern int fputc_unlocked (int __c, FILE *__stream);







extern int putc_unlocked (int __c, FILE *__stream);
extern int putchar_unlocked (int __c);






extern int getw (FILE *__stream);


extern int putw (int __w, FILE *__stream);








extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
     ;






extern char *gets (char *__s) ;

# 655 "/usr/include/stdio.h" 3





extern int fputs (__const char *__restrict __s, FILE *__restrict __stream);





extern int puts (__const char *__s);






extern int ungetc (int __c, FILE *__stream);






extern size_t fread (void *__restrict __ptr, size_t __size,
       size_t __n, FILE *__restrict __stream) ;




extern size_t fwrite (__const void *__restrict __ptr, size_t __size,
        size_t __n, FILE *__restrict __s) ;

# 708 "/usr/include/stdio.h" 3
extern size_t fread_unlocked (void *__restrict __ptr, size_t __size,
         size_t __n, FILE *__restrict __stream) ;
extern size_t fwrite_unlocked (__const void *__restrict __ptr, size_t __size,
          size_t __n, FILE *__restrict __stream) ;








extern int fseek (FILE *__stream, long int __off, int __whence);




extern long int ftell (FILE *__stream) ;




extern void rewind (FILE *__stream);

# 744 "/usr/include/stdio.h" 3
extern int fseeko (FILE *__stream, __off_t __off, int __whence);




extern __off_t ftello (FILE *__stream) ;
# 763 "/usr/include/stdio.h" 3






extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos);




extern int fsetpos (FILE *__stream, __const fpos_t *__pos);
# 786 "/usr/include/stdio.h" 3

# 795 "/usr/include/stdio.h" 3


extern void clearerr (FILE *__stream) __attribute__ ((__nothrow__));

extern int feof (FILE *__stream) __attribute__ ((__nothrow__)) ;

extern int ferror (FILE *__stream) __attribute__ ((__nothrow__)) ;




extern void clearerr_unlocked (FILE *__stream) __attribute__ ((__nothrow__));
extern int feof_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;
extern int ferror_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;








extern void perror (__const char *__s);






# 1 "/usr/include/bits/sys_errlist.h" 1 3
# 27 "/usr/include/bits/sys_errlist.h" 3
extern int sys_nerr;
extern __const char *__const sys_errlist[];
# 825 "/usr/include/stdio.h" 2 3




extern int fileno (FILE *__stream) __attribute__ ((__nothrow__)) ;




extern int fileno_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;
# 844 "/usr/include/stdio.h" 3
extern FILE *popen (__const char *__command, __const char *__modes) ;





extern int pclose (FILE *__stream);





extern char *ctermid (char *__s) __attribute__ ((__nothrow__));
# 884 "/usr/include/stdio.h" 3
extern void flockfile (FILE *__stream) __attribute__ ((__nothrow__));



extern int ftrylockfile (FILE *__stream) __attribute__ ((__nothrow__)) ;


extern void funlockfile (FILE *__stream) __attribute__ ((__nothrow__));
# 914 "/usr/include/stdio.h" 3

# 108 "exStbDemo.c" 2
# 1 "/usr/include/stdlib.h" 1 3
# 33 "/usr/include/stdlib.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 326 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 3 4
typedef long int wchar_t;
# 34 "/usr/include/stdlib.h" 2 3


# 96 "/usr/include/stdlib.h" 3


typedef struct
  {
    int quot;
    int rem;
  } div_t;



typedef struct
  {
    long int quot;
    long int rem;
  } ldiv_t;



# 140 "/usr/include/stdlib.h" 3
extern size_t __ctype_get_mb_cur_max (void) __attribute__ ((__nothrow__)) ;




extern double atof (__const char *__nptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern int atoi (__const char *__nptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern long int atol (__const char *__nptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;





__extension__ extern long long int atoll (__const char *__nptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;





extern double strtod (__const char *__restrict __nptr,
        char **__restrict __endptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;

# 182 "/usr/include/stdlib.h" 3


extern long int strtol (__const char *__restrict __nptr,
   char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;

extern unsigned long int strtoul (__const char *__restrict __nptr,
      char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;




__extension__
extern long long int strtoq (__const char *__restrict __nptr,
        char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;

__extension__
extern unsigned long long int strtouq (__const char *__restrict __nptr,
           char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;





__extension__
extern long long int strtoll (__const char *__restrict __nptr,
         char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;

__extension__
extern unsigned long long int strtoull (__const char *__restrict __nptr,
     char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;

# 311 "/usr/include/stdlib.h" 3
extern char *l64a (long int __n) __attribute__ ((__nothrow__)) ;


extern long int a64l (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;




# 1 "/usr/include/sys/types.h" 1 3
# 29 "/usr/include/sys/types.h" 3






typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;




typedef __loff_t loff_t;



typedef __ino_t ino_t;
# 62 "/usr/include/sys/types.h" 3
typedef __dev_t dev_t;




typedef __gid_t gid_t;




typedef __mode_t mode_t;




typedef __nlink_t nlink_t;




typedef __uid_t uid_t;





typedef __off_t off_t;
# 100 "/usr/include/sys/types.h" 3
typedef __pid_t pid_t;




typedef __id_t id_t;




typedef __ssize_t ssize_t;





typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;





typedef __key_t key_t;
# 133 "/usr/include/sys/types.h" 3
# 1 "/usr/include/time.h" 1 3
# 75 "/usr/include/time.h" 3


typedef __time_t time_t;



# 93 "/usr/include/time.h" 3
typedef __clockid_t clockid_t;
# 105 "/usr/include/time.h" 3
typedef __timer_t timer_t;
# 134 "/usr/include/sys/types.h" 2 3
# 147 "/usr/include/sys/types.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 148 "/usr/include/sys/types.h" 2 3



typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
# 195 "/usr/include/sys/types.h" 3
typedef int int8_t __attribute__ ((__mode__ (__QI__)));
typedef int int16_t __attribute__ ((__mode__ (__HI__)));
typedef int int32_t __attribute__ ((__mode__ (__SI__)));
typedef int int64_t __attribute__ ((__mode__ (__DI__)));


typedef unsigned int u_int8_t __attribute__ ((__mode__ (__QI__)));
typedef unsigned int u_int16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int u_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int u_int64_t __attribute__ ((__mode__ (__DI__)));

typedef int register_t __attribute__ ((__mode__ (__word__)));
# 217 "/usr/include/sys/types.h" 3
# 1 "/usr/include/endian.h" 1 3
# 37 "/usr/include/endian.h" 3
# 1 "/usr/include/bits/endian.h" 1 3
# 38 "/usr/include/endian.h" 2 3
# 61 "/usr/include/endian.h" 3
# 1 "/usr/include/bits/byteswap.h" 1 3
# 62 "/usr/include/endian.h" 2 3
# 218 "/usr/include/sys/types.h" 2 3


# 1 "/usr/include/sys/select.h" 1 3
# 31 "/usr/include/sys/select.h" 3
# 1 "/usr/include/bits/select.h" 1 3
# 32 "/usr/include/sys/select.h" 2 3


# 1 "/usr/include/bits/sigset.h" 1 3
# 24 "/usr/include/bits/sigset.h" 3
typedef int __sig_atomic_t;




typedef struct
  {
    unsigned long int __val[(1024 / (8 * sizeof (unsigned long int)))];
  } __sigset_t;
# 35 "/usr/include/sys/select.h" 2 3



typedef __sigset_t sigset_t;





# 1 "/usr/include/time.h" 1 3
# 121 "/usr/include/time.h" 3
struct timespec
  {
    __time_t tv_sec;
    long int tv_nsec;
  };
# 45 "/usr/include/sys/select.h" 2 3

# 1 "/usr/include/bits/time.h" 1 3
# 69 "/usr/include/bits/time.h" 3
struct timeval
  {
    __time_t tv_sec;
    __suseconds_t tv_usec;
  };
# 47 "/usr/include/sys/select.h" 2 3


typedef __suseconds_t suseconds_t;





typedef long int __fd_mask;
# 67 "/usr/include/sys/select.h" 3
typedef struct
  {






    __fd_mask __fds_bits[1024 / (8 * sizeof (__fd_mask))];


  } fd_set;






typedef __fd_mask fd_mask;
# 99 "/usr/include/sys/select.h" 3

# 109 "/usr/include/sys/select.h" 3
extern int select (int __nfds, fd_set *__restrict __readfds,
     fd_set *__restrict __writefds,
     fd_set *__restrict __exceptfds,
     struct timeval *__restrict __timeout);
# 121 "/usr/include/sys/select.h" 3
extern int pselect (int __nfds, fd_set *__restrict __readfds,
      fd_set *__restrict __writefds,
      fd_set *__restrict __exceptfds,
      const struct timespec *__restrict __timeout,
      const __sigset_t *__restrict __sigmask);



# 221 "/usr/include/sys/types.h" 2 3


# 1 "/usr/include/sys/sysmacros.h" 1 3
# 30 "/usr/include/sys/sysmacros.h" 3
__extension__
extern unsigned int gnu_dev_major (unsigned long long int __dev)
     __attribute__ ((__nothrow__));
__extension__
extern unsigned int gnu_dev_minor (unsigned long long int __dev)
     __attribute__ ((__nothrow__));
__extension__
extern unsigned long long int gnu_dev_makedev (unsigned int __major,
            unsigned int __minor)
     __attribute__ ((__nothrow__));
# 224 "/usr/include/sys/types.h" 2 3
# 235 "/usr/include/sys/types.h" 3
typedef __blkcnt_t blkcnt_t;



typedef __fsblkcnt_t fsblkcnt_t;



typedef __fsfilcnt_t fsfilcnt_t;
# 270 "/usr/include/sys/types.h" 3
# 1 "/usr/include/bits/pthreadtypes.h" 1 3
# 36 "/usr/include/bits/pthreadtypes.h" 3
typedef unsigned long int pthread_t;


typedef union
{
  char __size[36];
  long int __align;
} pthread_attr_t;


typedef struct __pthread_internal_slist
{
  struct __pthread_internal_slist *__next;
} __pthread_slist_t;




typedef union
{
  struct __pthread_mutex_s
  {
    int __lock;
    unsigned int __count;
    int __owner;


    int __kind;
    unsigned int __nusers;
    __extension__ union
    {
      int __spins;
      __pthread_slist_t __list;
    };
  } __data;
  char __size[24];
  long int __align;
} pthread_mutex_t;

typedef union
{
  char __size[4];
  long int __align;
} pthread_mutexattr_t;




typedef union
{
  struct
  {
    int __lock;
    unsigned int __futex;
    __extension__ unsigned long long int __total_seq;
    __extension__ unsigned long long int __wakeup_seq;
    __extension__ unsigned long long int __woken_seq;
    void *__mutex;
    unsigned int __nwaiters;
    unsigned int __broadcast_seq;
  } __data;
  char __size[48];
  __extension__ long long int __align;
} pthread_cond_t;

typedef union
{
  char __size[4];
  long int __align;
} pthread_condattr_t;



typedef unsigned int pthread_key_t;



typedef int pthread_once_t;





typedef union
{
  struct
  {
    int __lock;
    unsigned int __nr_readers;
    unsigned int __readers_wakeup;
    unsigned int __writer_wakeup;
    unsigned int __nr_readers_queued;
    unsigned int __nr_writers_queued;


    unsigned char __flags;
    unsigned char __shared;
    unsigned char __pad1;
    unsigned char __pad2;
    int __writer;
  } __data;
  char __size[32];
  long int __align;
} pthread_rwlock_t;

typedef union
{
  char __size[8];
  long int __align;
} pthread_rwlockattr_t;





typedef volatile int pthread_spinlock_t;




typedef union
{
  char __size[20];
  long int __align;
} pthread_barrier_t;

typedef union
{
  char __size[4];
  int __align;
} pthread_barrierattr_t;
# 271 "/usr/include/sys/types.h" 2 3



# 321 "/usr/include/stdlib.h" 2 3






extern long int random (void) __attribute__ ((__nothrow__));


extern void srandom (unsigned int __seed) __attribute__ ((__nothrow__));





extern char *initstate (unsigned int __seed, char *__statebuf,
   size_t __statelen) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));



extern char *setstate (char *__statebuf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));







struct random_data
  {
    int32_t *fptr;
    int32_t *rptr;
    int32_t *state;
    int rand_type;
    int rand_deg;
    int rand_sep;
    int32_t *end_ptr;
  };

extern int random_r (struct random_data *__restrict __buf,
       int32_t *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern int srandom_r (unsigned int __seed, struct random_data *__buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));

extern int initstate_r (unsigned int __seed, char *__restrict __statebuf,
   size_t __statelen,
   struct random_data *__restrict __buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 4)));

extern int setstate_r (char *__restrict __statebuf,
         struct random_data *__restrict __buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));






extern int rand (void) __attribute__ ((__nothrow__));

extern void srand (unsigned int __seed) __attribute__ ((__nothrow__));




extern int rand_r (unsigned int *__seed) __attribute__ ((__nothrow__));







extern double drand48 (void) __attribute__ ((__nothrow__));
extern double erand48 (unsigned short int __xsubi[3]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern long int lrand48 (void) __attribute__ ((__nothrow__));
extern long int nrand48 (unsigned short int __xsubi[3])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern long int mrand48 (void) __attribute__ ((__nothrow__));
extern long int jrand48 (unsigned short int __xsubi[3])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern void srand48 (long int __seedval) __attribute__ ((__nothrow__));
extern unsigned short int *seed48 (unsigned short int __seed16v[3])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern void lcong48 (unsigned short int __param[7]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





struct drand48_data
  {
    unsigned short int __x[3];
    unsigned short int __old_x[3];
    unsigned short int __c;
    unsigned short int __init;
    unsigned long long int __a;
  };


extern int drand48_r (struct drand48_data *__restrict __buffer,
        double *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern int erand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        double *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int lrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern int nrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int mrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern int jrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int srand48_r (long int __seedval, struct drand48_data *__buffer)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));

extern int seed48_r (unsigned short int __seed16v[3],
       struct drand48_data *__buffer) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern int lcong48_r (unsigned short int __param[7],
        struct drand48_data *__buffer)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));









extern void *malloc (size_t __size) __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;

extern void *calloc (size_t __nmemb, size_t __size)
     __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;










extern void *realloc (void *__ptr, size_t __size)
     __attribute__ ((__nothrow__)) __attribute__ ((__warn_unused_result__));

extern void free (void *__ptr) __attribute__ ((__nothrow__));




extern void cfree (void *__ptr) __attribute__ ((__nothrow__));



# 1 "/usr/include/alloca.h" 1 3
# 25 "/usr/include/alloca.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 26 "/usr/include/alloca.h" 2 3







extern void *alloca (size_t __size) __attribute__ ((__nothrow__));






# 498 "/usr/include/stdlib.h" 2 3




extern void *valloc (size_t __size) __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;




extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;




extern void abort (void) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));



extern int atexit (void (*__func) (void)) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int on_exit (void (*__func) (int __status, void *__arg), void *__arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern void exit (int __status) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));

# 543 "/usr/include/stdlib.h" 3


extern char *getenv (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;




extern char *__secure_getenv (__const char *__name)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;





extern int putenv (char *__string) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int setenv (__const char *__name, __const char *__value, int __replace)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));


extern int unsetenv (__const char *__name) __attribute__ ((__nothrow__));






extern int clearenv (void) __attribute__ ((__nothrow__));
# 583 "/usr/include/stdlib.h" 3
extern char *mktemp (char *__template) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 594 "/usr/include/stdlib.h" 3
extern int mkstemp (char *__template) __attribute__ ((__nonnull__ (1))) ;
# 614 "/usr/include/stdlib.h" 3
extern char *mkdtemp (char *__template) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 640 "/usr/include/stdlib.h" 3





extern int system (__const char *__command) ;

# 662 "/usr/include/stdlib.h" 3
extern char *realpath (__const char *__restrict __name,
         char *__restrict __resolved) __attribute__ ((__nothrow__)) ;






typedef int (*__compar_fn_t) (__const void *, __const void *);
# 680 "/usr/include/stdlib.h" 3



extern void *bsearch (__const void *__key, __const void *__base,
        size_t __nmemb, size_t __size, __compar_fn_t __compar)
     __attribute__ ((__nonnull__ (1, 2, 5))) ;



extern void qsort (void *__base, size_t __nmemb, size_t __size,
     __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 4)));
# 699 "/usr/include/stdlib.h" 3
extern int abs (int __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;
extern long int labs (long int __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;












extern div_t div (int __numer, int __denom)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;
extern ldiv_t ldiv (long int __numer, long int __denom)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;

# 735 "/usr/include/stdlib.h" 3
extern char *ecvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4))) ;




extern char *fcvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4))) ;




extern char *gcvt (double __value, int __ndigit, char *__buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3))) ;




extern char *qecvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qfcvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qgcvt (long double __value, int __ndigit, char *__buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3))) ;




extern int ecvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int fcvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));

extern int qecvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int qfcvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));







extern int mblen (__const char *__s, size_t __n) __attribute__ ((__nothrow__)) ;


extern int mbtowc (wchar_t *__restrict __pwc,
     __const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__)) ;


extern int wctomb (char *__s, wchar_t __wchar) __attribute__ ((__nothrow__)) ;



extern size_t mbstowcs (wchar_t *__restrict __pwcs,
   __const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__));

extern size_t wcstombs (char *__restrict __s,
   __const wchar_t *__restrict __pwcs, size_t __n)
     __attribute__ ((__nothrow__));








extern int rpmatch (__const char *__response) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 840 "/usr/include/stdlib.h" 3
extern int posix_openpt (int __oflag) ;
# 875 "/usr/include/stdlib.h" 3
extern int getloadavg (double __loadavg[], int __nelem)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 891 "/usr/include/stdlib.h" 3

# 109 "exStbDemo.c" 2
# 1 "/usr/include/string.h" 1 3
# 28 "/usr/include/string.h" 3





# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 34 "/usr/include/string.h" 2 3




extern void *memcpy (void *__restrict __dest,
       __const void *__restrict __src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern void *memmove (void *__dest, __const void *__src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));






extern void *memccpy (void *__restrict __dest, __const void *__restrict __src,
        int __c, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));





extern void *memset (void *__s, int __c, size_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int memcmp (__const void *__s1, __const void *__s2, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern void *memchr (__const void *__s, int __c, size_t __n)
      __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

# 82 "/usr/include/string.h" 3


extern char *strcpy (char *__restrict __dest, __const char *__restrict __src)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern char *strncpy (char *__restrict __dest,
        __const char *__restrict __src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern char *strcat (char *__restrict __dest, __const char *__restrict __src)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern char *strncat (char *__restrict __dest, __const char *__restrict __src,
        size_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strcmp (__const char *__s1, __const char *__s2)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern int strncmp (__const char *__s1, __const char *__s2, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strcoll (__const char *__s1, __const char *__s2)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern size_t strxfrm (char *__restrict __dest,
         __const char *__restrict __src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));

# 130 "/usr/include/string.h" 3
extern char *strdup (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) __attribute__ ((__nonnull__ (1)));
# 165 "/usr/include/string.h" 3


extern char *strchr (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

extern char *strrchr (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

# 181 "/usr/include/string.h" 3



extern size_t strcspn (__const char *__s, __const char *__reject)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern size_t strspn (__const char *__s, __const char *__accept)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern char *strpbrk (__const char *__s, __const char *__accept)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern char *strstr (__const char *__haystack, __const char *__needle)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));



extern char *strtok (char *__restrict __s, __const char *__restrict __delim)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));




extern char *__strtok_r (char *__restrict __s,
    __const char *__restrict __delim,
    char **__restrict __save_ptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 3)));

extern char *strtok_r (char *__restrict __s, __const char *__restrict __delim,
         char **__restrict __save_ptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 3)));
# 240 "/usr/include/string.h" 3


extern size_t strlen (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

# 254 "/usr/include/string.h" 3


extern char *strerror (int __errnum) __attribute__ ((__nothrow__));

# 270 "/usr/include/string.h" 3
extern int strerror_r (int __errnum, char *__buf, size_t __buflen) __asm__ ("" "__xpg_strerror_r") __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));
# 294 "/usr/include/string.h" 3
extern void __bzero (void *__s, size_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern void bcopy (__const void *__src, void *__dest, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern void bzero (void *__s, size_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int bcmp (__const void *__s1, __const void *__s2, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern char *index (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));


extern char *rindex (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));



extern int ffs (int __i) __attribute__ ((__nothrow__)) __attribute__ ((__const__));
# 331 "/usr/include/string.h" 3
extern int strcasecmp (__const char *__s1, __const char *__s2)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strncasecmp (__const char *__s1, __const char *__s2, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
# 354 "/usr/include/string.h" 3
extern char *strsep (char **__restrict __stringp,
       __const char *__restrict __delim)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
# 432 "/usr/include/string.h" 3

# 110 "exStbDemo.c" 2
# 1 "/usr/include/unistd.h" 1 3
# 28 "/usr/include/unistd.h" 3

# 173 "/usr/include/unistd.h" 3
# 1 "/usr/include/bits/posix_opt.h" 1 3
# 174 "/usr/include/unistd.h" 2 3
# 197 "/usr/include/unistd.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 198 "/usr/include/unistd.h" 2 3
# 226 "/usr/include/unistd.h" 3
typedef __useconds_t useconds_t;
# 238 "/usr/include/unistd.h" 3
typedef __intptr_t intptr_t;






typedef __socklen_t socklen_t;
# 258 "/usr/include/unistd.h" 3
extern int access (__const char *__name, int __type) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 301 "/usr/include/unistd.h" 3
extern __off_t lseek (int __fd, __off_t __offset, int __whence) __attribute__ ((__nothrow__));
# 320 "/usr/include/unistd.h" 3
extern int close (int __fd);






extern ssize_t read (int __fd, void *__buf, size_t __nbytes) ;





extern ssize_t write (int __fd, __const void *__buf, size_t __n) ;
# 384 "/usr/include/unistd.h" 3
extern int pipe (int __pipedes[2]) __attribute__ ((__nothrow__)) ;
# 399 "/usr/include/unistd.h" 3
extern unsigned int alarm (unsigned int __seconds) __attribute__ ((__nothrow__));
# 411 "/usr/include/unistd.h" 3
extern unsigned int sleep (unsigned int __seconds);






extern __useconds_t ualarm (__useconds_t __value, __useconds_t __interval)
     __attribute__ ((__nothrow__));






extern int usleep (__useconds_t __useconds);
# 435 "/usr/include/unistd.h" 3
extern int pause (void);



extern int chown (__const char *__file, __uid_t __owner, __gid_t __group)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern int fchown (int __fd, __uid_t __owner, __gid_t __group) __attribute__ ((__nothrow__)) ;




extern int lchown (__const char *__file, __uid_t __owner, __gid_t __group)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 463 "/usr/include/unistd.h" 3
extern int chdir (__const char *__path) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern int fchdir (int __fd) __attribute__ ((__nothrow__)) ;
# 477 "/usr/include/unistd.h" 3
extern char *getcwd (char *__buf, size_t __size) __attribute__ ((__nothrow__)) ;
# 490 "/usr/include/unistd.h" 3
extern char *getwd (char *__buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__deprecated__)) ;




extern int dup (int __fd) __attribute__ ((__nothrow__)) ;


extern int dup2 (int __fd, int __fd2) __attribute__ ((__nothrow__));
# 508 "/usr/include/unistd.h" 3
extern char **__environ;







extern int execve (__const char *__path, char *__const __argv[],
     char *__const __envp[]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 528 "/usr/include/unistd.h" 3
extern int execv (__const char *__path, char *__const __argv[])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int execle (__const char *__path, __const char *__arg, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int execl (__const char *__path, __const char *__arg, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int execvp (__const char *__file, char *__const __argv[])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));




extern int execlp (__const char *__file, __const char *__arg, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));




extern int nice (int __inc) __attribute__ ((__nothrow__)) ;




extern void _exit (int __status) __attribute__ ((__noreturn__));





# 1 "/usr/include/bits/confname.h" 1 3
# 26 "/usr/include/bits/confname.h" 3
enum
  {
    _PC_LINK_MAX,

    _PC_MAX_CANON,

    _PC_MAX_INPUT,

    _PC_NAME_MAX,

    _PC_PATH_MAX,

    _PC_PIPE_BUF,

    _PC_CHOWN_RESTRICTED,

    _PC_NO_TRUNC,

    _PC_VDISABLE,

    _PC_SYNC_IO,

    _PC_ASYNC_IO,

    _PC_PRIO_IO,

    _PC_SOCK_MAXBUF,

    _PC_FILESIZEBITS,

    _PC_REC_INCR_XFER_SIZE,

    _PC_REC_MAX_XFER_SIZE,

    _PC_REC_MIN_XFER_SIZE,

    _PC_REC_XFER_ALIGN,

    _PC_ALLOC_SIZE_MIN,

    _PC_SYMLINK_MAX,

    _PC_2_SYMLINKS

  };


enum
  {
    _SC_ARG_MAX,

    _SC_CHILD_MAX,

    _SC_CLK_TCK,

    _SC_NGROUPS_MAX,

    _SC_OPEN_MAX,

    _SC_STREAM_MAX,

    _SC_TZNAME_MAX,

    _SC_JOB_CONTROL,

    _SC_SAVED_IDS,

    _SC_REALTIME_SIGNALS,

    _SC_PRIORITY_SCHEDULING,

    _SC_TIMERS,

    _SC_ASYNCHRONOUS_IO,

    _SC_PRIORITIZED_IO,

    _SC_SYNCHRONIZED_IO,

    _SC_FSYNC,

    _SC_MAPPED_FILES,

    _SC_MEMLOCK,

    _SC_MEMLOCK_RANGE,

    _SC_MEMORY_PROTECTION,

    _SC_MESSAGE_PASSING,

    _SC_SEMAPHORES,

    _SC_SHARED_MEMORY_OBJECTS,

    _SC_AIO_LISTIO_MAX,

    _SC_AIO_MAX,

    _SC_AIO_PRIO_DELTA_MAX,

    _SC_DELAYTIMER_MAX,

    _SC_MQ_OPEN_MAX,

    _SC_MQ_PRIO_MAX,

    _SC_VERSION,

    _SC_PAGESIZE,


    _SC_RTSIG_MAX,

    _SC_SEM_NSEMS_MAX,

    _SC_SEM_VALUE_MAX,

    _SC_SIGQUEUE_MAX,

    _SC_TIMER_MAX,




    _SC_BC_BASE_MAX,

    _SC_BC_DIM_MAX,

    _SC_BC_SCALE_MAX,

    _SC_BC_STRING_MAX,

    _SC_COLL_WEIGHTS_MAX,

    _SC_EQUIV_CLASS_MAX,

    _SC_EXPR_NEST_MAX,

    _SC_LINE_MAX,

    _SC_RE_DUP_MAX,

    _SC_CHARCLASS_NAME_MAX,


    _SC_2_VERSION,

    _SC_2_C_BIND,

    _SC_2_C_DEV,

    _SC_2_FORT_DEV,

    _SC_2_FORT_RUN,

    _SC_2_SW_DEV,

    _SC_2_LOCALEDEF,


    _SC_PII,

    _SC_PII_XTI,

    _SC_PII_SOCKET,

    _SC_PII_INTERNET,

    _SC_PII_OSI,

    _SC_POLL,

    _SC_SELECT,

    _SC_UIO_MAXIOV,

    _SC_IOV_MAX = _SC_UIO_MAXIOV,

    _SC_PII_INTERNET_STREAM,

    _SC_PII_INTERNET_DGRAM,

    _SC_PII_OSI_COTS,

    _SC_PII_OSI_CLTS,

    _SC_PII_OSI_M,

    _SC_T_IOV_MAX,



    _SC_THREADS,

    _SC_THREAD_SAFE_FUNCTIONS,

    _SC_GETGR_R_SIZE_MAX,

    _SC_GETPW_R_SIZE_MAX,

    _SC_LOGIN_NAME_MAX,

    _SC_TTY_NAME_MAX,

    _SC_THREAD_DESTRUCTOR_ITERATIONS,

    _SC_THREAD_KEYS_MAX,

    _SC_THREAD_STACK_MIN,

    _SC_THREAD_THREADS_MAX,

    _SC_THREAD_ATTR_STACKADDR,

    _SC_THREAD_ATTR_STACKSIZE,

    _SC_THREAD_PRIORITY_SCHEDULING,

    _SC_THREAD_PRIO_INHERIT,

    _SC_THREAD_PRIO_PROTECT,

    _SC_THREAD_PROCESS_SHARED,


    _SC_NPROCESSORS_CONF,

    _SC_NPROCESSORS_ONLN,

    _SC_PHYS_PAGES,

    _SC_AVPHYS_PAGES,

    _SC_ATEXIT_MAX,

    _SC_PASS_MAX,


    _SC_XOPEN_VERSION,

    _SC_XOPEN_XCU_VERSION,

    _SC_XOPEN_UNIX,

    _SC_XOPEN_CRYPT,

    _SC_XOPEN_ENH_I18N,

    _SC_XOPEN_SHM,


    _SC_2_CHAR_TERM,

    _SC_2_C_VERSION,

    _SC_2_UPE,


    _SC_XOPEN_XPG2,

    _SC_XOPEN_XPG3,

    _SC_XOPEN_XPG4,


    _SC_CHAR_BIT,

    _SC_CHAR_MAX,

    _SC_CHAR_MIN,

    _SC_INT_MAX,

    _SC_INT_MIN,

    _SC_LONG_BIT,

    _SC_WORD_BIT,

    _SC_MB_LEN_MAX,

    _SC_NZERO,

    _SC_SSIZE_MAX,

    _SC_SCHAR_MAX,

    _SC_SCHAR_MIN,

    _SC_SHRT_MAX,

    _SC_SHRT_MIN,

    _SC_UCHAR_MAX,

    _SC_UINT_MAX,

    _SC_ULONG_MAX,

    _SC_USHRT_MAX,


    _SC_NL_ARGMAX,

    _SC_NL_LANGMAX,

    _SC_NL_MSGMAX,

    _SC_NL_NMAX,

    _SC_NL_SETMAX,

    _SC_NL_TEXTMAX,


    _SC_XBS5_ILP32_OFF32,

    _SC_XBS5_ILP32_OFFBIG,

    _SC_XBS5_LP64_OFF64,

    _SC_XBS5_LPBIG_OFFBIG,


    _SC_XOPEN_LEGACY,

    _SC_XOPEN_REALTIME,

    _SC_XOPEN_REALTIME_THREADS,


    _SC_ADVISORY_INFO,

    _SC_BARRIERS,

    _SC_BASE,

    _SC_C_LANG_SUPPORT,

    _SC_C_LANG_SUPPORT_R,

    _SC_CLOCK_SELECTION,

    _SC_CPUTIME,

    _SC_THREAD_CPUTIME,

    _SC_DEVICE_IO,

    _SC_DEVICE_SPECIFIC,

    _SC_DEVICE_SPECIFIC_R,

    _SC_FD_MGMT,

    _SC_FIFO,

    _SC_PIPE,

    _SC_FILE_ATTRIBUTES,

    _SC_FILE_LOCKING,

    _SC_FILE_SYSTEM,

    _SC_MONOTONIC_CLOCK,

    _SC_MULTI_PROCESS,

    _SC_SINGLE_PROCESS,

    _SC_NETWORKING,

    _SC_READER_WRITER_LOCKS,

    _SC_SPIN_LOCKS,

    _SC_REGEXP,

    _SC_REGEX_VERSION,

    _SC_SHELL,

    _SC_SIGNALS,

    _SC_SPAWN,

    _SC_SPORADIC_SERVER,

    _SC_THREAD_SPORADIC_SERVER,

    _SC_SYSTEM_DATABASE,

    _SC_SYSTEM_DATABASE_R,

    _SC_TIMEOUTS,

    _SC_TYPED_MEMORY_OBJECTS,

    _SC_USER_GROUPS,

    _SC_USER_GROUPS_R,

    _SC_2_PBS,

    _SC_2_PBS_ACCOUNTING,

    _SC_2_PBS_LOCATE,

    _SC_2_PBS_MESSAGE,

    _SC_2_PBS_TRACK,

    _SC_SYMLOOP_MAX,

    _SC_STREAMS,

    _SC_2_PBS_CHECKPOINT,


    _SC_V6_ILP32_OFF32,

    _SC_V6_ILP32_OFFBIG,

    _SC_V6_LP64_OFF64,

    _SC_V6_LPBIG_OFFBIG,


    _SC_HOST_NAME_MAX,

    _SC_TRACE,

    _SC_TRACE_EVENT_FILTER,

    _SC_TRACE_INHERIT,

    _SC_TRACE_LOG,


    _SC_LEVEL1_ICACHE_SIZE,

    _SC_LEVEL1_ICACHE_ASSOC,

    _SC_LEVEL1_ICACHE_LINESIZE,

    _SC_LEVEL1_DCACHE_SIZE,

    _SC_LEVEL1_DCACHE_ASSOC,

    _SC_LEVEL1_DCACHE_LINESIZE,

    _SC_LEVEL2_CACHE_SIZE,

    _SC_LEVEL2_CACHE_ASSOC,

    _SC_LEVEL2_CACHE_LINESIZE,

    _SC_LEVEL3_CACHE_SIZE,

    _SC_LEVEL3_CACHE_ASSOC,

    _SC_LEVEL3_CACHE_LINESIZE,

    _SC_LEVEL4_CACHE_SIZE,

    _SC_LEVEL4_CACHE_ASSOC,

    _SC_LEVEL4_CACHE_LINESIZE,



    _SC_IPV6 = _SC_LEVEL1_ICACHE_SIZE + 50,

    _SC_RAW_SOCKETS

  };


enum
  {
    _CS_PATH,


    _CS_V6_WIDTH_RESTRICTED_ENVS,



    _CS_GNU_LIBC_VERSION,

    _CS_GNU_LIBPTHREAD_VERSION,


    _CS_LFS_CFLAGS = 1000,

    _CS_LFS_LDFLAGS,

    _CS_LFS_LIBS,

    _CS_LFS_LINTFLAGS,

    _CS_LFS64_CFLAGS,

    _CS_LFS64_LDFLAGS,

    _CS_LFS64_LIBS,

    _CS_LFS64_LINTFLAGS,


    _CS_XBS5_ILP32_OFF32_CFLAGS = 1100,

    _CS_XBS5_ILP32_OFF32_LDFLAGS,

    _CS_XBS5_ILP32_OFF32_LIBS,

    _CS_XBS5_ILP32_OFF32_LINTFLAGS,

    _CS_XBS5_ILP32_OFFBIG_CFLAGS,

    _CS_XBS5_ILP32_OFFBIG_LDFLAGS,

    _CS_XBS5_ILP32_OFFBIG_LIBS,

    _CS_XBS5_ILP32_OFFBIG_LINTFLAGS,

    _CS_XBS5_LP64_OFF64_CFLAGS,

    _CS_XBS5_LP64_OFF64_LDFLAGS,

    _CS_XBS5_LP64_OFF64_LIBS,

    _CS_XBS5_LP64_OFF64_LINTFLAGS,

    _CS_XBS5_LPBIG_OFFBIG_CFLAGS,

    _CS_XBS5_LPBIG_OFFBIG_LDFLAGS,

    _CS_XBS5_LPBIG_OFFBIG_LIBS,

    _CS_XBS5_LPBIG_OFFBIG_LINTFLAGS,


    _CS_POSIX_V6_ILP32_OFF32_CFLAGS,

    _CS_POSIX_V6_ILP32_OFF32_LDFLAGS,

    _CS_POSIX_V6_ILP32_OFF32_LIBS,

    _CS_POSIX_V6_ILP32_OFF32_LINTFLAGS,

    _CS_POSIX_V6_ILP32_OFFBIG_CFLAGS,

    _CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS,

    _CS_POSIX_V6_ILP32_OFFBIG_LIBS,

    _CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS,

    _CS_POSIX_V6_LP64_OFF64_CFLAGS,

    _CS_POSIX_V6_LP64_OFF64_LDFLAGS,

    _CS_POSIX_V6_LP64_OFF64_LIBS,

    _CS_POSIX_V6_LP64_OFF64_LINTFLAGS,

    _CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS,

    _CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS,

    _CS_POSIX_V6_LPBIG_OFFBIG_LIBS,

    _CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS

  };
# 567 "/usr/include/unistd.h" 2 3


extern long int pathconf (__const char *__path, int __name)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern long int fpathconf (int __fd, int __name) __attribute__ ((__nothrow__));


extern long int sysconf (int __name) __attribute__ ((__nothrow__));



extern size_t confstr (int __name, char *__buf, size_t __len) __attribute__ ((__nothrow__));




extern __pid_t getpid (void) __attribute__ ((__nothrow__));


extern __pid_t getppid (void) __attribute__ ((__nothrow__));




extern __pid_t getpgrp (void) __attribute__ ((__nothrow__));
# 603 "/usr/include/unistd.h" 3
extern __pid_t __getpgid (__pid_t __pid) __attribute__ ((__nothrow__));
# 612 "/usr/include/unistd.h" 3
extern int setpgid (__pid_t __pid, __pid_t __pgid) __attribute__ ((__nothrow__));
# 629 "/usr/include/unistd.h" 3
extern int setpgrp (void) __attribute__ ((__nothrow__));
# 646 "/usr/include/unistd.h" 3
extern __pid_t setsid (void) __attribute__ ((__nothrow__));







extern __uid_t getuid (void) __attribute__ ((__nothrow__));


extern __uid_t geteuid (void) __attribute__ ((__nothrow__));


extern __gid_t getgid (void) __attribute__ ((__nothrow__));


extern __gid_t getegid (void) __attribute__ ((__nothrow__));




extern int getgroups (int __size, __gid_t __list[]) __attribute__ ((__nothrow__)) ;
# 679 "/usr/include/unistd.h" 3
extern int setuid (__uid_t __uid) __attribute__ ((__nothrow__));




extern int setreuid (__uid_t __ruid, __uid_t __euid) __attribute__ ((__nothrow__));




extern int seteuid (__uid_t __uid) __attribute__ ((__nothrow__));






extern int setgid (__gid_t __gid) __attribute__ ((__nothrow__));




extern int setregid (__gid_t __rgid, __gid_t __egid) __attribute__ ((__nothrow__));




extern int setegid (__gid_t __gid) __attribute__ ((__nothrow__));
# 735 "/usr/include/unistd.h" 3
extern __pid_t fork (void) __attribute__ ((__nothrow__));






extern __pid_t vfork (void) __attribute__ ((__nothrow__));





extern char *ttyname (int __fd) __attribute__ ((__nothrow__));



extern int ttyname_r (int __fd, char *__buf, size_t __buflen)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2))) ;



extern int isatty (int __fd) __attribute__ ((__nothrow__));





extern int ttyslot (void) __attribute__ ((__nothrow__));




extern int link (__const char *__from, __const char *__to)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;
# 781 "/usr/include/unistd.h" 3
extern int symlink (__const char *__from, __const char *__to)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;




extern ssize_t readlink (__const char *__restrict __path,
    char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;
# 804 "/usr/include/unistd.h" 3
extern int unlink (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 813 "/usr/include/unistd.h" 3
extern int rmdir (__const char *__path) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern __pid_t tcgetpgrp (int __fd) __attribute__ ((__nothrow__));


extern int tcsetpgrp (int __fd, __pid_t __pgrp_id) __attribute__ ((__nothrow__));






extern char *getlogin (void);







extern int getlogin_r (char *__name, size_t __name_len) __attribute__ ((__nonnull__ (1)));




extern int setlogin (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 849 "/usr/include/unistd.h" 3
# 1 "/usr/include/getopt.h" 1 3
# 59 "/usr/include/getopt.h" 3
extern char *optarg;
# 73 "/usr/include/getopt.h" 3
extern int optind;




extern int opterr;



extern int optopt;
# 152 "/usr/include/getopt.h" 3
extern int getopt (int ___argc, char *const *___argv, const char *__shortopts)
       __attribute__ ((__nothrow__));
# 850 "/usr/include/unistd.h" 2 3







extern int gethostname (char *__name, size_t __len) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern int sethostname (__const char *__name, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern int sethostid (long int __id) __attribute__ ((__nothrow__)) ;





extern int getdomainname (char *__name, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
extern int setdomainname (__const char *__name, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;





extern int vhangup (void) __attribute__ ((__nothrow__));


extern int revoke (__const char *__file) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;







extern int profil (unsigned short int *__sample_buffer, size_t __size,
     size_t __offset, unsigned int __scale)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int acct (__const char *__name) __attribute__ ((__nothrow__));



extern char *getusershell (void) __attribute__ ((__nothrow__));
extern void endusershell (void) __attribute__ ((__nothrow__));
extern void setusershell (void) __attribute__ ((__nothrow__));





extern int daemon (int __nochdir, int __noclose) __attribute__ ((__nothrow__)) ;






extern int chroot (__const char *__path) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern char *getpass (__const char *__prompt) __attribute__ ((__nonnull__ (1)));
# 935 "/usr/include/unistd.h" 3
extern int fsync (int __fd);






extern long int gethostid (void);


extern void sync (void) __attribute__ ((__nothrow__));




extern int getpagesize (void) __attribute__ ((__nothrow__)) __attribute__ ((__const__));




extern int getdtablesize (void) __attribute__ ((__nothrow__));




extern int truncate (__const char *__file, __off_t __length)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 982 "/usr/include/unistd.h" 3
extern int ftruncate (int __fd, __off_t __length) __attribute__ ((__nothrow__)) ;
# 1002 "/usr/include/unistd.h" 3
extern int brk (void *__addr) __attribute__ ((__nothrow__)) ;





extern void *sbrk (intptr_t __delta) __attribute__ ((__nothrow__));
# 1023 "/usr/include/unistd.h" 3
extern long int syscall (long int __sysno, ...) __attribute__ ((__nothrow__));
# 1046 "/usr/include/unistd.h" 3
extern int lockf (int __fd, int __cmd, __off_t __len) ;
# 1077 "/usr/include/unistd.h" 3
extern int fdatasync (int __fildes);
# 1115 "/usr/include/unistd.h" 3

# 111 "exStbDemo.c" 2
# 1 "/usr/include/errno.h" 1 3
# 32 "/usr/include/errno.h" 3




# 1 "/usr/include/bits/errno.h" 1 3
# 25 "/usr/include/bits/errno.h" 3
# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/errno.h" 1 3



# 1 "/usr/include/asm/errno.h" 1 3



# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/asm-generic/errno.h" 1 3



# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/asm-generic/errno-base.h" 1 3
# 5 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/asm-generic/errno.h" 2 3
# 5 "/usr/include/asm/errno.h" 2 3
# 5 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/errno.h" 2 3
# 26 "/usr/include/bits/errno.h" 2 3
# 43 "/usr/include/bits/errno.h" 3
extern int *__errno_location (void) __attribute__ ((__nothrow__)) __attribute__ ((__const__));
# 37 "/usr/include/errno.h" 2 3
# 59 "/usr/include/errno.h" 3

# 112 "exStbDemo.c" 2
# 1 "/usr/include/signal.h" 1 3
# 31 "/usr/include/signal.h" 3


# 1 "/usr/include/bits/sigset.h" 1 3
# 104 "/usr/include/bits/sigset.h" 3
extern int __sigismember (__const __sigset_t *, int);
extern int __sigaddset (__sigset_t *, int);
extern int __sigdelset (__sigset_t *, int);
# 34 "/usr/include/signal.h" 2 3







typedef __sig_atomic_t sig_atomic_t;

# 58 "/usr/include/signal.h" 3
# 1 "/usr/include/bits/signum.h" 1 3
# 59 "/usr/include/signal.h" 2 3
# 75 "/usr/include/signal.h" 3
typedef void (*__sighandler_t) (int);




extern __sighandler_t __sysv_signal (int __sig, __sighandler_t __handler)
     __attribute__ ((__nothrow__));
# 90 "/usr/include/signal.h" 3


extern __sighandler_t signal (int __sig, __sighandler_t __handler)
     __attribute__ ((__nothrow__));
# 104 "/usr/include/signal.h" 3

# 117 "/usr/include/signal.h" 3
extern int kill (__pid_t __pid, int __sig) __attribute__ ((__nothrow__));






extern int killpg (__pid_t __pgrp, int __sig) __attribute__ ((__nothrow__));




extern int raise (int __sig) __attribute__ ((__nothrow__));




extern __sighandler_t ssignal (int __sig, __sighandler_t __handler)
     __attribute__ ((__nothrow__));
extern int gsignal (int __sig) __attribute__ ((__nothrow__));




extern void psignal (int __sig, __const char *__s);
# 153 "/usr/include/signal.h" 3
extern int __sigpause (int __sig_or_mask, int __is_sig);
# 181 "/usr/include/signal.h" 3
extern int sigblock (int __mask) __attribute__ ((__nothrow__)) __attribute__ ((__deprecated__));


extern int sigsetmask (int __mask) __attribute__ ((__nothrow__)) __attribute__ ((__deprecated__));


extern int siggetmask (void) __attribute__ ((__nothrow__)) __attribute__ ((__deprecated__));
# 201 "/usr/include/signal.h" 3
typedef __sighandler_t sig_t;







# 1 "/usr/include/time.h" 1 3
# 210 "/usr/include/signal.h" 2 3


# 1 "/usr/include/bits/siginfo.h" 1 3
# 25 "/usr/include/bits/siginfo.h" 3
# 1 "/usr/include/bits/wordsize.h" 1 3
# 26 "/usr/include/bits/siginfo.h" 2 3







typedef union sigval
  {
    int sival_int;
    void *sival_ptr;
  } sigval_t;
# 51 "/usr/include/bits/siginfo.h" 3
typedef struct siginfo
  {
    int si_signo;
    int si_errno;

    int si_code;

    union
      {
 int _pad[((128 / sizeof (int)) - 3)];


 struct
   {
     __pid_t si_pid;
     __uid_t si_uid;
   } _kill;


 struct
   {
     int si_tid;
     int si_overrun;
     sigval_t si_sigval;
   } _timer;


 struct
   {
     __pid_t si_pid;
     __uid_t si_uid;
     sigval_t si_sigval;
   } _rt;


 struct
   {
     __pid_t si_pid;
     __uid_t si_uid;
     int si_status;
     __clock_t si_utime;
     __clock_t si_stime;
   } _sigchld;


 struct
   {
     void *si_addr;
   } _sigfault;


 struct
   {
     long int si_band;
     int si_fd;
   } _sigpoll;
      } _sifields;
  } siginfo_t;
# 129 "/usr/include/bits/siginfo.h" 3
enum
{
  SI_ASYNCNL = -60,

  SI_TKILL = -6,

  SI_SIGIO,

  SI_ASYNCIO,

  SI_MESGQ,

  SI_TIMER,

  SI_QUEUE,

  SI_USER,

  SI_KERNEL = 0x80

};



enum
{
  ILL_ILLOPC = 1,

  ILL_ILLOPN,

  ILL_ILLADR,

  ILL_ILLTRP,

  ILL_PRVOPC,

  ILL_PRVREG,

  ILL_COPROC,

  ILL_BADSTK

};


enum
{
  FPE_INTDIV = 1,

  FPE_INTOVF,

  FPE_FLTDIV,

  FPE_FLTOVF,

  FPE_FLTUND,

  FPE_FLTRES,

  FPE_FLTINV,

  FPE_FLTSUB

};


enum
{
  SEGV_MAPERR = 1,

  SEGV_ACCERR

};


enum
{
  BUS_ADRALN = 1,

  BUS_ADRERR,

  BUS_OBJERR

};


enum
{
  TRAP_BRKPT = 1,

  TRAP_TRACE

};


enum
{
  CLD_EXITED = 1,

  CLD_KILLED,

  CLD_DUMPED,

  CLD_TRAPPED,

  CLD_STOPPED,

  CLD_CONTINUED

};


enum
{
  POLL_IN = 1,

  POLL_OUT,

  POLL_MSG,

  POLL_ERR,

  POLL_PRI,

  POLL_HUP

};
# 273 "/usr/include/bits/siginfo.h" 3
typedef struct sigevent
  {
    sigval_t sigev_value;
    int sigev_signo;
    int sigev_notify;

    union
      {
 int _pad[((64 / sizeof (int)) - 3)];



 __pid_t _tid;

 struct
   {
     void (*_function) (sigval_t);
     void *_attribute;
   } _sigev_thread;
      } _sigev_un;
  } sigevent_t;






enum
{
  SIGEV_SIGNAL = 0,

  SIGEV_NONE,

  SIGEV_THREAD,


  SIGEV_THREAD_ID = 4

};
# 213 "/usr/include/signal.h" 2 3



extern int sigemptyset (sigset_t *__set) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int sigfillset (sigset_t *__set) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int sigaddset (sigset_t *__set, int __signo) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int sigdelset (sigset_t *__set, int __signo) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int sigismember (__const sigset_t *__set, int __signo)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 246 "/usr/include/signal.h" 3
# 1 "/usr/include/bits/sigaction.h" 1 3
# 25 "/usr/include/bits/sigaction.h" 3
struct sigaction
  {


    union
      {

 __sighandler_t sa_handler;

 void (*sa_sigaction) (int, siginfo_t *, void *);
      }
    __sigaction_handler;







    __sigset_t sa_mask;


    int sa_flags;


    void (*sa_restorer) (void);
  };
# 247 "/usr/include/signal.h" 2 3


extern int sigprocmask (int __how, __const sigset_t *__restrict __set,
   sigset_t *__restrict __oset) __attribute__ ((__nothrow__));






extern int sigsuspend (__const sigset_t *__set) __attribute__ ((__nonnull__ (1)));


extern int sigaction (int __sig, __const struct sigaction *__restrict __act,
        struct sigaction *__restrict __oact) __attribute__ ((__nothrow__));


extern int sigpending (sigset_t *__set) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern int sigwait (__const sigset_t *__restrict __set, int *__restrict __sig)
     __attribute__ ((__nonnull__ (1, 2)));






extern int sigwaitinfo (__const sigset_t *__restrict __set,
   siginfo_t *__restrict __info) __attribute__ ((__nonnull__ (1)));






extern int sigtimedwait (__const sigset_t *__restrict __set,
    siginfo_t *__restrict __info,
    __const struct timespec *__restrict __timeout)
     __attribute__ ((__nonnull__ (1)));



extern int sigqueue (__pid_t __pid, int __sig, __const union sigval __val)
     __attribute__ ((__nothrow__));
# 304 "/usr/include/signal.h" 3
extern __const char *__const _sys_siglist[65];
extern __const char *__const sys_siglist[65];


struct sigvec
  {
    __sighandler_t sv_handler;
    int sv_mask;

    int sv_flags;

  };
# 328 "/usr/include/signal.h" 3
extern int sigvec (int __sig, __const struct sigvec *__vec,
     struct sigvec *__ovec) __attribute__ ((__nothrow__));



# 1 "/usr/include/bits/sigcontext.h" 1 3
# 28 "/usr/include/bits/sigcontext.h" 3
# 1 "/usr/include/asm/sigcontext.h" 1 3
# 19 "/usr/include/asm/sigcontext.h" 3
struct _fpreg {
 unsigned short significand[4];
 unsigned short exponent;
};

struct _fpxreg {
 unsigned short significand[4];
 unsigned short exponent;
 unsigned short padding[3];
};

struct _xmmreg {
 unsigned long element[4];
};

struct _fpstate {

 unsigned long cw;
 unsigned long sw;
 unsigned long tag;
 unsigned long ipoff;
 unsigned long cssel;
 unsigned long dataoff;
 unsigned long datasel;
 struct _fpreg _st[8];
 unsigned short status;
 unsigned short magic;


 unsigned long _fxsr_env[6];
 unsigned long mxcsr;
 unsigned long reserved;
 struct _fpxreg _fxsr_st[8];
 struct _xmmreg _xmm[8];
 unsigned long padding[56];
};



struct sigcontext {
 unsigned short gs, __gsh;
 unsigned short fs, __fsh;
 unsigned short es, __esh;
 unsigned short ds, __dsh;
 unsigned long edi;
 unsigned long esi;
 unsigned long ebp;
 unsigned long esp;
 unsigned long ebx;
 unsigned long edx;
 unsigned long ecx;
 unsigned long eax;
 unsigned long trapno;
 unsigned long err;
 unsigned long eip;
 unsigned short cs, __csh;
 unsigned long eflags;
 unsigned long esp_at_signal;
 unsigned short ss, __ssh;
 struct _fpstate * fpstate;
 unsigned long oldmask;
 unsigned long cr2;
};
# 29 "/usr/include/bits/sigcontext.h" 2 3
# 334 "/usr/include/signal.h" 2 3


extern int sigreturn (struct sigcontext *__scp) __attribute__ ((__nothrow__));






# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 344 "/usr/include/signal.h" 2 3




extern int siginterrupt (int __sig, int __interrupt) __attribute__ ((__nothrow__));

# 1 "/usr/include/bits/sigstack.h" 1 3
# 26 "/usr/include/bits/sigstack.h" 3
struct sigstack
  {
    void *ss_sp;
    int ss_onstack;
  };



enum
{
  SS_ONSTACK = 1,

  SS_DISABLE

};
# 50 "/usr/include/bits/sigstack.h" 3
typedef struct sigaltstack
  {
    void *ss_sp;
    int ss_flags;
    size_t ss_size;
  } stack_t;
# 351 "/usr/include/signal.h" 2 3
# 359 "/usr/include/signal.h" 3
extern int sigstack (struct sigstack *__ss, struct sigstack *__oss)
     __attribute__ ((__nothrow__)) __attribute__ ((__deprecated__));



extern int sigaltstack (__const struct sigaltstack *__restrict __ss,
   struct sigaltstack *__restrict __oss) __attribute__ ((__nothrow__));
# 389 "/usr/include/signal.h" 3
# 1 "/usr/include/bits/sigthread.h" 1 3
# 31 "/usr/include/bits/sigthread.h" 3
extern int pthread_sigmask (int __how,
       __const __sigset_t *__restrict __newmask,
       __sigset_t *__restrict __oldmask)__attribute__ ((__nothrow__));


extern int pthread_kill (pthread_t __threadid, int __signo) __attribute__ ((__nothrow__));
# 390 "/usr/include/signal.h" 2 3






extern int __libc_current_sigrtmin (void) __attribute__ ((__nothrow__));

extern int __libc_current_sigrtmax (void) __attribute__ ((__nothrow__));




# 113 "exStbDemo.c" 2


# 1 "/usr/include/sys/stat.h" 1 3
# 39 "/usr/include/sys/stat.h" 3
# 1 "/usr/include/time.h" 1 3
# 40 "/usr/include/sys/stat.h" 2 3
# 105 "/usr/include/sys/stat.h" 3


# 1 "/usr/include/bits/stat.h" 1 3
# 36 "/usr/include/bits/stat.h" 3
struct stat
  {
    __dev_t st_dev;
    unsigned short int __pad1;

    __ino_t st_ino;



    __mode_t st_mode;
    __nlink_t st_nlink;
    __uid_t st_uid;
    __gid_t st_gid;
    __dev_t st_rdev;
    unsigned short int __pad2;

    __off_t st_size;



    __blksize_t st_blksize;


    __blkcnt_t st_blocks;
# 70 "/usr/include/bits/stat.h" 3
    struct timespec st_atim;
    struct timespec st_mtim;
    struct timespec st_ctim;
# 85 "/usr/include/bits/stat.h" 3
    unsigned long int __unused4;
    unsigned long int __unused5;



  };
# 108 "/usr/include/sys/stat.h" 2 3
# 209 "/usr/include/sys/stat.h" 3
extern int stat (__const char *__restrict __file,
   struct stat *__restrict __buf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern int fstat (int __fd, struct stat *__buf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));
# 261 "/usr/include/sys/stat.h" 3
extern int lstat (__const char *__restrict __file,
    struct stat *__restrict __buf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
# 282 "/usr/include/sys/stat.h" 3
extern int chmod (__const char *__file, __mode_t __mode)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int lchmod (__const char *__file, __mode_t __mode)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));




extern int fchmod (int __fd, __mode_t __mode) __attribute__ ((__nothrow__));
# 309 "/usr/include/sys/stat.h" 3
extern __mode_t umask (__mode_t __mask) __attribute__ ((__nothrow__));
# 318 "/usr/include/sys/stat.h" 3
extern int mkdir (__const char *__path, __mode_t __mode)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 333 "/usr/include/sys/stat.h" 3
extern int mknod (__const char *__path, __mode_t __mode, __dev_t __dev)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 347 "/usr/include/sys/stat.h" 3
extern int mkfifo (__const char *__path, __mode_t __mode)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 397 "/usr/include/sys/stat.h" 3
extern int __fxstat (int __ver, int __fildes, struct stat *__stat_buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3)));
extern int __xstat (int __ver, __const char *__filename,
      struct stat *__stat_buf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 3)));
extern int __lxstat (int __ver, __const char *__filename,
       struct stat *__stat_buf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 3)));
extern int __fxstatat (int __ver, int __fildes, __const char *__filename,
         struct stat *__stat_buf, int __flag)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4)));
# 440 "/usr/include/sys/stat.h" 3
extern int __xmknod (int __ver, __const char *__path, __mode_t __mode,
       __dev_t *__dev) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 4)));

extern int __xmknodat (int __ver, int __fd, __const char *__path,
         __mode_t __mode, __dev_t *__dev)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 5)));
# 532 "/usr/include/sys/stat.h" 3

# 116 "exStbDemo.c" 2
# 1 "/usr/include/sys/ioctl.h" 1 3
# 24 "/usr/include/sys/ioctl.h" 3



# 1 "/usr/include/bits/ioctls.h" 1 3
# 24 "/usr/include/bits/ioctls.h" 3
# 1 "/usr/include/asm/ioctls.h" 1 3



# 1 "/usr/include/asm/ioctl.h" 1 3
# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/asm-generic/ioctl.h" 1 3
# 51 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/asm-generic/ioctl.h" 3
extern unsigned int __invalid_size_argument_for_IOC;
# 1 "/usr/include/asm/ioctl.h" 2 3
# 5 "/usr/include/asm/ioctls.h" 2 3
# 25 "/usr/include/bits/ioctls.h" 2 3
# 28 "/usr/include/sys/ioctl.h" 2 3


# 1 "/usr/include/bits/ioctl-types.h" 1 3
# 28 "/usr/include/bits/ioctl-types.h" 3
struct winsize
  {
    unsigned short int ws_row;
    unsigned short int ws_col;
    unsigned short int ws_xpixel;
    unsigned short int ws_ypixel;
  };


struct termio
  {
    unsigned short int c_iflag;
    unsigned short int c_oflag;
    unsigned short int c_cflag;
    unsigned short int c_lflag;
    unsigned char c_line;
    unsigned char c_cc[8];
};
# 31 "/usr/include/sys/ioctl.h" 2 3






# 1 "/usr/include/sys/ttydefaults.h" 1 3
# 38 "/usr/include/sys/ioctl.h" 2 3




extern int ioctl (int __fd, unsigned long int __request, ...) __attribute__ ((__nothrow__));


# 117 "exStbDemo.c" 2
# 1 "/usr/include/sys/time.h" 1 3
# 27 "/usr/include/sys/time.h" 3
# 1 "/usr/include/time.h" 1 3
# 28 "/usr/include/sys/time.h" 2 3

# 1 "/usr/include/bits/time.h" 1 3
# 30 "/usr/include/sys/time.h" 2 3
# 39 "/usr/include/sys/time.h" 3

# 57 "/usr/include/sys/time.h" 3
struct timezone
  {
    int tz_minuteswest;
    int tz_dsttime;
  };

typedef struct timezone *__restrict __timezone_ptr_t;
# 73 "/usr/include/sys/time.h" 3
extern int gettimeofday (struct timeval *__restrict __tv,
    __timezone_ptr_t __tz) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));




extern int settimeofday (__const struct timeval *__tv,
    __const struct timezone *__tz)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int adjtime (__const struct timeval *__delta,
      struct timeval *__olddelta) __attribute__ ((__nothrow__));




enum __itimer_which
  {

    ITIMER_REAL = 0,


    ITIMER_VIRTUAL = 1,



    ITIMER_PROF = 2

  };



struct itimerval
  {

    struct timeval it_interval;

    struct timeval it_value;
  };






typedef int __itimer_which_t;




extern int getitimer (__itimer_which_t __which,
        struct itimerval *__value) __attribute__ ((__nothrow__));




extern int setitimer (__itimer_which_t __which,
        __const struct itimerval *__restrict __new,
        struct itimerval *__restrict __old) __attribute__ ((__nothrow__));




extern int utimes (__const char *__file, __const struct timeval __tvp[2])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int lutimes (__const char *__file, __const struct timeval __tvp[2])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int futimes (int __fd, __const struct timeval __tvp[2]) __attribute__ ((__nothrow__));
# 191 "/usr/include/sys/time.h" 3

# 118 "exStbDemo.c" 2

# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/fb.h" 1



# 1 "/usr/include/asm/types.h" 1 3





typedef unsigned short umode_t;






typedef __signed__ char __s8;
typedef unsigned char __u8;

typedef __signed__ short __s16;
typedef unsigned short __u16;

typedef __signed__ int __s32;
typedef unsigned int __u32;


typedef __signed__ long long __s64;
typedef unsigned long long __u64;
# 5 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/fb.h" 2
# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/i2c.h" 1
# 29 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/i2c.h"
# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/types.h" 1
# 11 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/types.h"
# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/posix_types.h" 1



# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/stddef.h" 1



# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/compiler.h" 1
# 5 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/stddef.h" 2
# 5 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/posix_types.h" 2
# 36 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/posix_types.h"
typedef struct {
 unsigned long fds_bits [(1024/(8 * sizeof(unsigned long)))];
} __kernel_fd_set;


typedef void (*__kernel_sighandler_t)(int);


typedef int __kernel_key_t;
typedef int __kernel_mqd_t;

# 1 "/usr/include/asm/posix_types.h" 1 3
# 10 "/usr/include/asm/posix_types.h" 3
typedef unsigned long __kernel_ino_t;
typedef unsigned short __kernel_mode_t;
typedef unsigned short __kernel_nlink_t;
typedef long __kernel_off_t;
typedef int __kernel_pid_t;
typedef unsigned short __kernel_ipc_pid_t;
typedef unsigned short __kernel_uid_t;
typedef unsigned short __kernel_gid_t;
typedef unsigned int __kernel_size_t;
typedef int __kernel_ssize_t;
typedef int __kernel_ptrdiff_t;
typedef long __kernel_time_t;
typedef long __kernel_suseconds_t;
typedef long __kernel_clock_t;
typedef int __kernel_timer_t;
typedef int __kernel_clockid_t;
typedef int __kernel_daddr_t;
typedef char * __kernel_caddr_t;
typedef unsigned short __kernel_uid16_t;
typedef unsigned short __kernel_gid16_t;
typedef unsigned int __kernel_uid32_t;
typedef unsigned int __kernel_gid32_t;

typedef unsigned short __kernel_old_uid_t;
typedef unsigned short __kernel_old_gid_t;
typedef unsigned short __kernel_old_dev_t;


typedef long long __kernel_loff_t;


typedef struct {



 int __val[2];

} __kernel_fsid_t;
# 48 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/posix_types.h" 2
# 12 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/types.h" 2
# 180 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/types.h"
typedef __u16 __le16;
typedef __u16 __be16;
typedef __u32 __le32;
typedef __u32 __be32;

typedef __u64 __le64;
typedef __u64 __be64;

typedef __u16 __sum16;
typedef __u32 __wsum;
# 202 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/types.h"
struct ustat {
 __kernel_daddr_t f_tfree;
 __kernel_ino_t f_tinode;
 char f_fname[6];
 char f_fpack[6];
};
# 30 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/i2c.h" 2
# 470 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/i2c.h"
struct i2c_msg {
 __u16 addr;
 __u16 flags;







 __u16 len;
 __u8 *buf;
};
# 532 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/i2c.h"
union i2c_smbus_data {
 __u8 byte;
 __u16 word;
 __u8 block[32 + 2];

};
# 6 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/fb.h" 2

struct dentry;
# 149 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/fb.h"
struct fb_fix_screeninfo {
 char id[16];
 unsigned long smem_start;

 __u32 smem_len;
 __u32 type;
 __u32 type_aux;
 __u32 visual;
 __u16 xpanstep;
 __u16 ypanstep;
 __u16 ywrapstep;
 __u32 line_length;
 unsigned long mmio_start;

 __u32 mmio_len;
 __u32 accel;

 __u16 reserved[3];
};







struct fb_bitfield {
 __u32 offset;
 __u32 length;
 __u32 msb_right;

};
# 228 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/fb.h"
struct fb_var_screeninfo {
 __u32 xres;
 __u32 yres;
 __u32 xres_virtual;
 __u32 yres_virtual;
 __u32 xoffset;
 __u32 yoffset;

 __u32 bits_per_pixel;
 __u32 grayscale;

 struct fb_bitfield red;
 struct fb_bitfield green;
 struct fb_bitfield blue;
 struct fb_bitfield transp;

 __u32 nonstd;

 __u32 activate;

 __u32 height;
 __u32 width;

 __u32 accel_flags;


 __u32 pixclock;
 __u32 left_margin;
 __u32 right_margin;
 __u32 upper_margin;
 __u32 lower_margin;
 __u32 hsync_len;
 __u32 vsync_len;
 __u32 sync;
 __u32 vmode;
 __u32 rotate;
 __u32 reserved[5];
};

struct fb_cmap {
 __u32 start;
 __u32 len;
 __u16 *red;
 __u16 *green;
 __u16 *blue;
 __u16 *transp;
};

struct fb_con2fbmap {
 __u32 console;
 __u32 framebuffer;
};
# 288 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/fb.h"
enum {

 FB_BLANK_UNBLANK = 0,


 FB_BLANK_NORMAL = 0 + 1,


 FB_BLANK_VSYNC_SUSPEND = 1 + 1,


 FB_BLANK_HSYNC_SUSPEND = 2 + 1,


 FB_BLANK_POWERDOWN = 3 + 1
};
# 315 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/fb.h"
struct fb_vblank {
 __u32 flags;
 __u32 count;
 __u32 vcount;
 __u32 hcount;
 __u32 reserved[4];
};





struct fb_copyarea {
 __u32 dx;
 __u32 dy;
 __u32 width;
 __u32 height;
 __u32 sx;
 __u32 sy;
};

struct fb_fillrect {
 __u32 dx;
 __u32 dy;
 __u32 width;
 __u32 height;
 __u32 color;
 __u32 rop;
};

struct fb_image {
 __u32 dx;
 __u32 dy;
 __u32 width;
 __u32 height;
 __u32 fg_color;
 __u32 bg_color;
 __u8 depth;
 const char *data;
 struct fb_cmap cmap;
};
# 369 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/fb.h"
struct fbcurpos {
 __u16 x, y;
};

struct fb_cursor {
 __u16 set;
 __u16 enable;
 __u16 rop;
 const char *mask;
 struct fbcurpos hot;
 struct fb_image image;
};
# 120 "exStbDemo.c" 2

# 1 "/usr/include/sys/prctl.h" 1 3
# 23 "/usr/include/sys/prctl.h" 3
# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/prctl.h" 1 3
# 24 "/usr/include/sys/prctl.h" 2 3




extern int prctl (int __option, ...) __attribute__ ((__nothrow__));


# 122 "exStbDemo.c" 2

# 1 "/usr/include/fcntl.h" 1 3
# 30 "/usr/include/fcntl.h" 3




# 1 "/usr/include/bits/fcntl.h" 1 3
# 144 "/usr/include/bits/fcntl.h" 3
struct flock
  {
    short int l_type;
    short int l_whence;

    __off_t l_start;
    __off_t l_len;




    __pid_t l_pid;
  };
# 211 "/usr/include/bits/fcntl.h" 3

# 240 "/usr/include/bits/fcntl.h" 3

# 35 "/usr/include/fcntl.h" 2 3
# 76 "/usr/include/fcntl.h" 3
extern int fcntl (int __fd, int __cmd, ...);
# 85 "/usr/include/fcntl.h" 3
extern int open (__const char *__file, int __oflag, ...) __attribute__ ((__nonnull__ (1)));
# 130 "/usr/include/fcntl.h" 3
extern int creat (__const char *__file, __mode_t __mode) __attribute__ ((__nonnull__ (1)));
# 176 "/usr/include/fcntl.h" 3
extern int posix_fadvise (int __fd, __off_t __offset, __off_t __len,
     int __advise) __attribute__ ((__nothrow__));
# 198 "/usr/include/fcntl.h" 3
extern int posix_fallocate (int __fd, __off_t __offset, __off_t __len);
# 220 "/usr/include/fcntl.h" 3

# 124 "exStbDemo.c" 2

# 1 "/usr/include/pthread.h" 1 3
# 25 "/usr/include/pthread.h" 3
# 1 "/usr/include/sched.h" 1 3
# 29 "/usr/include/sched.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 30 "/usr/include/sched.h" 2 3


# 1 "/usr/include/time.h" 1 3
# 33 "/usr/include/sched.h" 2 3


# 1 "/usr/include/bits/sched.h" 1 3
# 71 "/usr/include/bits/sched.h" 3
struct sched_param
  {
    int __sched_priority;
  };





extern int clone (int (*__fn) (void *__arg), void *__child_stack,
    int __flags, void *__arg, ...) __attribute__ ((__nothrow__));


extern int unshare (int __flags) __attribute__ ((__nothrow__));


extern int sched_getcpu (void) __attribute__ ((__nothrow__));










struct __sched_param
  {
    int __sched_priority;
  };
# 113 "/usr/include/bits/sched.h" 3
typedef unsigned long int __cpu_mask;






typedef struct
{
  __cpu_mask __bits[1024 / (8 * sizeof (__cpu_mask))];
} cpu_set_t;
# 196 "/usr/include/bits/sched.h" 3


extern int __sched_cpucount (size_t __setsize, const cpu_set_t *__setp)
  __attribute__ ((__nothrow__));
extern cpu_set_t *__sched_cpualloc (size_t __count) __attribute__ ((__nothrow__)) ;
extern void __sched_cpufree (cpu_set_t *__set) __attribute__ ((__nothrow__));


# 36 "/usr/include/sched.h" 2 3







extern int sched_setparam (__pid_t __pid, __const struct sched_param *__param)
     __attribute__ ((__nothrow__));


extern int sched_getparam (__pid_t __pid, struct sched_param *__param) __attribute__ ((__nothrow__));


extern int sched_setscheduler (__pid_t __pid, int __policy,
          __const struct sched_param *__param) __attribute__ ((__nothrow__));


extern int sched_getscheduler (__pid_t __pid) __attribute__ ((__nothrow__));


extern int sched_yield (void) __attribute__ ((__nothrow__));


extern int sched_get_priority_max (int __algorithm) __attribute__ ((__nothrow__));


extern int sched_get_priority_min (int __algorithm) __attribute__ ((__nothrow__));


extern int sched_rr_get_interval (__pid_t __pid, struct timespec *__t) __attribute__ ((__nothrow__));
# 118 "/usr/include/sched.h" 3

# 26 "/usr/include/pthread.h" 2 3
# 1 "/usr/include/time.h" 1 3
# 31 "/usr/include/time.h" 3








# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 40 "/usr/include/time.h" 2 3



# 1 "/usr/include/bits/time.h" 1 3
# 44 "/usr/include/time.h" 2 3
# 59 "/usr/include/time.h" 3


typedef __clock_t clock_t;



# 132 "/usr/include/time.h" 3


struct tm
{
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;


  long int tm_gmtoff;
  __const char *tm_zone;




};








struct itimerspec
  {
    struct timespec it_interval;
    struct timespec it_value;
  };


struct sigevent;
# 181 "/usr/include/time.h" 3



extern clock_t clock (void) __attribute__ ((__nothrow__));


extern time_t time (time_t *__timer) __attribute__ ((__nothrow__));


extern double difftime (time_t __time1, time_t __time0)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern time_t mktime (struct tm *__tp) __attribute__ ((__nothrow__));





extern size_t strftime (char *__restrict __s, size_t __maxsize,
   __const char *__restrict __format,
   __const struct tm *__restrict __tp) __attribute__ ((__nothrow__));

# 229 "/usr/include/time.h" 3



extern struct tm *gmtime (__const time_t *__timer) __attribute__ ((__nothrow__));



extern struct tm *localtime (__const time_t *__timer) __attribute__ ((__nothrow__));





extern struct tm *gmtime_r (__const time_t *__restrict __timer,
       struct tm *__restrict __tp) __attribute__ ((__nothrow__));



extern struct tm *localtime_r (__const time_t *__restrict __timer,
          struct tm *__restrict __tp) __attribute__ ((__nothrow__));





extern char *asctime (__const struct tm *__tp) __attribute__ ((__nothrow__));


extern char *ctime (__const time_t *__timer) __attribute__ ((__nothrow__));







extern char *asctime_r (__const struct tm *__restrict __tp,
   char *__restrict __buf) __attribute__ ((__nothrow__));


extern char *ctime_r (__const time_t *__restrict __timer,
        char *__restrict __buf) __attribute__ ((__nothrow__));




extern char *__tzname[2];
extern int __daylight;
extern long int __timezone;




extern char *tzname[2];



extern void tzset (void) __attribute__ ((__nothrow__));



extern int daylight;
extern long int timezone;





extern int stime (__const time_t *__when) __attribute__ ((__nothrow__));
# 312 "/usr/include/time.h" 3
extern time_t timegm (struct tm *__tp) __attribute__ ((__nothrow__));


extern time_t timelocal (struct tm *__tp) __attribute__ ((__nothrow__));


extern int dysize (int __year) __attribute__ ((__nothrow__)) __attribute__ ((__const__));
# 327 "/usr/include/time.h" 3
extern int nanosleep (__const struct timespec *__requested_time,
        struct timespec *__remaining);



extern int clock_getres (clockid_t __clock_id, struct timespec *__res) __attribute__ ((__nothrow__));


extern int clock_gettime (clockid_t __clock_id, struct timespec *__tp) __attribute__ ((__nothrow__));


extern int clock_settime (clockid_t __clock_id, __const struct timespec *__tp)
     __attribute__ ((__nothrow__));






extern int clock_nanosleep (clockid_t __clock_id, int __flags,
       __const struct timespec *__req,
       struct timespec *__rem);


extern int clock_getcpuclockid (pid_t __pid, clockid_t *__clock_id) __attribute__ ((__nothrow__));




extern int timer_create (clockid_t __clock_id,
    struct sigevent *__restrict __evp,
    timer_t *__restrict __timerid) __attribute__ ((__nothrow__));


extern int timer_delete (timer_t __timerid) __attribute__ ((__nothrow__));


extern int timer_settime (timer_t __timerid, int __flags,
     __const struct itimerspec *__restrict __value,
     struct itimerspec *__restrict __ovalue) __attribute__ ((__nothrow__));


extern int timer_gettime (timer_t __timerid, struct itimerspec *__value)
     __attribute__ ((__nothrow__));


extern int timer_getoverrun (timer_t __timerid) __attribute__ ((__nothrow__));
# 416 "/usr/include/time.h" 3

# 27 "/usr/include/pthread.h" 2 3




# 1 "/usr/include/bits/setjmp.h" 1 3
# 29 "/usr/include/bits/setjmp.h" 3
typedef int __jmp_buf[6];
# 32 "/usr/include/pthread.h" 2 3
# 1 "/usr/include/bits/wordsize.h" 1 3
# 33 "/usr/include/pthread.h" 2 3



enum
{
  PTHREAD_CREATE_JOINABLE,

  PTHREAD_CREATE_DETACHED

};



enum
{
  PTHREAD_MUTEX_TIMED_NP,
  PTHREAD_MUTEX_RECURSIVE_NP,
  PTHREAD_MUTEX_ERRORCHECK_NP,
  PTHREAD_MUTEX_ADAPTIVE_NP
# 63 "/usr/include/pthread.h" 3
};
# 115 "/usr/include/pthread.h" 3
enum
{
  PTHREAD_RWLOCK_PREFER_READER_NP,
  PTHREAD_RWLOCK_PREFER_WRITER_NP,
  PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP,
  PTHREAD_RWLOCK_DEFAULT_NP = PTHREAD_RWLOCK_PREFER_READER_NP
};
# 147 "/usr/include/pthread.h" 3
enum
{
  PTHREAD_INHERIT_SCHED,

  PTHREAD_EXPLICIT_SCHED

};



enum
{
  PTHREAD_SCOPE_SYSTEM,

  PTHREAD_SCOPE_PROCESS

};



enum
{
  PTHREAD_PROCESS_PRIVATE,

  PTHREAD_PROCESS_SHARED

};
# 182 "/usr/include/pthread.h" 3
struct _pthread_cleanup_buffer
{
  void (*__routine) (void *);
  void *__arg;
  int __canceltype;
  struct _pthread_cleanup_buffer *__prev;
};


enum
{
  PTHREAD_CANCEL_ENABLE,

  PTHREAD_CANCEL_DISABLE

};
enum
{
  PTHREAD_CANCEL_DEFERRED,

  PTHREAD_CANCEL_ASYNCHRONOUS

};
# 220 "/usr/include/pthread.h" 3





extern int pthread_create (pthread_t *__restrict __newthread,
      __const pthread_attr_t *__restrict __attr,
      void *(*__start_routine) (void *),
      void *__restrict __arg) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 3)));





extern void pthread_exit (void *__retval) __attribute__ ((__noreturn__));







extern int pthread_join (pthread_t __th, void **__thread_return);
# 263 "/usr/include/pthread.h" 3
extern int pthread_detach (pthread_t __th) __attribute__ ((__nothrow__));



extern pthread_t pthread_self (void) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern int pthread_equal (pthread_t __thread1, pthread_t __thread2) __attribute__ ((__nothrow__));







extern int pthread_attr_init (pthread_attr_t *__attr) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_attr_destroy (pthread_attr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_attr_getdetachstate (__const pthread_attr_t *__attr,
     int *__detachstate)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setdetachstate (pthread_attr_t *__attr,
     int __detachstate)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_attr_getguardsize (__const pthread_attr_t *__attr,
          size_t *__guardsize)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setguardsize (pthread_attr_t *__attr,
          size_t __guardsize)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_attr_getschedparam (__const pthread_attr_t *__restrict
           __attr,
           struct sched_param *__restrict __param)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setschedparam (pthread_attr_t *__restrict __attr,
           __const struct sched_param *__restrict
           __param) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_getschedpolicy (__const pthread_attr_t *__restrict
     __attr, int *__restrict __policy)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setschedpolicy (pthread_attr_t *__attr, int __policy)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_attr_getinheritsched (__const pthread_attr_t *__restrict
      __attr, int *__restrict __inherit)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setinheritsched (pthread_attr_t *__attr,
      int __inherit)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_attr_getscope (__const pthread_attr_t *__restrict __attr,
      int *__restrict __scope)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setscope (pthread_attr_t *__attr, int __scope)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_attr_getstackaddr (__const pthread_attr_t *__restrict
          __attr, void **__restrict __stackaddr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) __attribute__ ((__deprecated__));





extern int pthread_attr_setstackaddr (pthread_attr_t *__attr,
          void *__stackaddr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__deprecated__));


extern int pthread_attr_getstacksize (__const pthread_attr_t *__restrict
          __attr, size_t *__restrict __stacksize)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));




extern int pthread_attr_setstacksize (pthread_attr_t *__attr,
          size_t __stacksize)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_attr_getstack (__const pthread_attr_t *__restrict __attr,
      void **__restrict __stackaddr,
      size_t *__restrict __stacksize)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2, 3)));




extern int pthread_attr_setstack (pthread_attr_t *__attr, void *__stackaddr,
      size_t __stacksize) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 413 "/usr/include/pthread.h" 3
extern int pthread_setschedparam (pthread_t __target_thread, int __policy,
      __const struct sched_param *__param)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3)));


extern int pthread_getschedparam (pthread_t __target_thread,
      int *__restrict __policy,
      struct sched_param *__restrict __param)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 3)));


extern int pthread_setschedprio (pthread_t __target_thread, int __prio)
     __attribute__ ((__nothrow__));
# 466 "/usr/include/pthread.h" 3
extern int pthread_once (pthread_once_t *__once_control,
    void (*__init_routine) (void)) __attribute__ ((__nonnull__ (1, 2)));
# 478 "/usr/include/pthread.h" 3
extern int pthread_setcancelstate (int __state, int *__oldstate);



extern int pthread_setcanceltype (int __type, int *__oldtype);


extern int pthread_cancel (pthread_t __th);




extern void pthread_testcancel (void);




typedef struct
{
  struct
  {
    __jmp_buf __cancel_jmp_buf;
    int __mask_was_saved;
  } __cancel_jmp_buf[1];
  void *__pad[4];
} __pthread_unwind_buf_t __attribute__ ((__aligned__));
# 512 "/usr/include/pthread.h" 3
struct __pthread_cleanup_frame
{
  void (*__cancel_routine) (void *);
  void *__cancel_arg;
  int __do_it;
  int __cancel_type;
};
# 652 "/usr/include/pthread.h" 3
extern void __pthread_register_cancel (__pthread_unwind_buf_t *__buf)
     __attribute__ ((__regparm__ (1)));
# 664 "/usr/include/pthread.h" 3
extern void __pthread_unregister_cancel (__pthread_unwind_buf_t *__buf)
  __attribute__ ((__regparm__ (1)));
# 705 "/usr/include/pthread.h" 3
extern void __pthread_unwind_next (__pthread_unwind_buf_t *__buf)
     __attribute__ ((__regparm__ (1))) __attribute__ ((__noreturn__))

     __attribute__ ((__weak__))

     ;



struct __jmp_buf_tag;
extern int __sigsetjmp (struct __jmp_buf_tag *__env, int __savemask) __attribute__ ((__nothrow__));





extern int pthread_mutex_init (pthread_mutex_t *__mutex,
          __const pthread_mutexattr_t *__mutexattr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_mutex_destroy (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_mutex_trylock (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_mutex_lock (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_mutex_timedlock (pthread_mutex_t *__restrict __mutex,
                                    __const struct timespec *__restrict
                                    __abstime) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern int pthread_mutex_unlock (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 776 "/usr/include/pthread.h" 3
extern int pthread_mutexattr_init (pthread_mutexattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_mutexattr_destroy (pthread_mutexattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_mutexattr_getpshared (__const pthread_mutexattr_t *
      __restrict __attr,
      int *__restrict __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_mutexattr_setpshared (pthread_mutexattr_t *__attr,
      int __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 848 "/usr/include/pthread.h" 3
extern int pthread_rwlock_init (pthread_rwlock_t *__restrict __rwlock,
    __const pthread_rwlockattr_t *__restrict
    __attr) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlock_destroy (pthread_rwlock_t *__rwlock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlock_rdlock (pthread_rwlock_t *__rwlock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlock_tryrdlock (pthread_rwlock_t *__rwlock)
  __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_rwlock_timedrdlock (pthread_rwlock_t *__restrict __rwlock,
           __const struct timespec *__restrict
           __abstime) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern int pthread_rwlock_wrlock (pthread_rwlock_t *__rwlock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlock_trywrlock (pthread_rwlock_t *__rwlock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_rwlock_timedwrlock (pthread_rwlock_t *__restrict __rwlock,
           __const struct timespec *__restrict
           __abstime) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern int pthread_rwlock_unlock (pthread_rwlock_t *__rwlock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int pthread_rwlockattr_init (pthread_rwlockattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlockattr_destroy (pthread_rwlockattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlockattr_getpshared (__const pthread_rwlockattr_t *
       __restrict __attr,
       int *__restrict __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_rwlockattr_setpshared (pthread_rwlockattr_t *__attr,
       int __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlockattr_getkind_np (__const pthread_rwlockattr_t *
       __restrict __attr,
       int *__restrict __pref)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_rwlockattr_setkind_np (pthread_rwlockattr_t *__attr,
       int __pref) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));







extern int pthread_cond_init (pthread_cond_t *__restrict __cond,
         __const pthread_condattr_t *__restrict
         __cond_attr) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_cond_destroy (pthread_cond_t *__cond)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_cond_signal (pthread_cond_t *__cond)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_cond_broadcast (pthread_cond_t *__cond)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern int pthread_cond_wait (pthread_cond_t *__restrict __cond,
         pthread_mutex_t *__restrict __mutex)
     __attribute__ ((__nonnull__ (1, 2)));
# 960 "/usr/include/pthread.h" 3
extern int pthread_cond_timedwait (pthread_cond_t *__restrict __cond,
       pthread_mutex_t *__restrict __mutex,
       __const struct timespec *__restrict
       __abstime) __attribute__ ((__nonnull__ (1, 2, 3)));




extern int pthread_condattr_init (pthread_condattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_condattr_destroy (pthread_condattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_condattr_getpshared (__const pthread_condattr_t *
                                        __restrict __attr,
                                        int *__restrict __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_condattr_setpshared (pthread_condattr_t *__attr,
                                        int __pshared) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_condattr_getclock (__const pthread_condattr_t *
          __restrict __attr,
          __clockid_t *__restrict __clock_id)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_condattr_setclock (pthread_condattr_t *__attr,
          __clockid_t __clock_id)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 1004 "/usr/include/pthread.h" 3
extern int pthread_spin_init (pthread_spinlock_t *__lock, int __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_spin_destroy (pthread_spinlock_t *__lock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_spin_lock (pthread_spinlock_t *__lock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_spin_trylock (pthread_spinlock_t *__lock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_spin_unlock (pthread_spinlock_t *__lock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern int pthread_barrier_init (pthread_barrier_t *__restrict __barrier,
     __const pthread_barrierattr_t *__restrict
     __attr, unsigned int __count)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_barrier_destroy (pthread_barrier_t *__barrier)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_barrier_wait (pthread_barrier_t *__barrier)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_barrierattr_init (pthread_barrierattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_barrierattr_destroy (pthread_barrierattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_barrierattr_getpshared (__const pthread_barrierattr_t *
        __restrict __attr,
        int *__restrict __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_barrierattr_setpshared (pthread_barrierattr_t *__attr,
                                           int __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 1071 "/usr/include/pthread.h" 3
extern int pthread_key_create (pthread_key_t *__key,
          void (*__destr_function) (void *))
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_key_delete (pthread_key_t __key) __attribute__ ((__nothrow__));


extern void *pthread_getspecific (pthread_key_t __key) __attribute__ ((__nothrow__));


extern int pthread_setspecific (pthread_key_t __key,
    __const void *__pointer) __attribute__ ((__nothrow__)) ;




extern int pthread_getcpuclockid (pthread_t __thread_id,
      __clockid_t *__clock_id)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));
# 1105 "/usr/include/pthread.h" 3
extern int pthread_atfork (void (*__prepare) (void),
      void (*__parent) (void),
      void (*__child) (void)) __attribute__ ((__nothrow__));
# 1119 "/usr/include/pthread.h" 3

# 126 "exStbDemo.c" 2
# 1 "/usr/include/ctype.h" 1 3
# 30 "/usr/include/ctype.h" 3

# 48 "/usr/include/ctype.h" 3
enum
{
  _ISupper = ((0) < 8 ? ((1 << (0)) << 8) : ((1 << (0)) >> 8)),
  _ISlower = ((1) < 8 ? ((1 << (1)) << 8) : ((1 << (1)) >> 8)),
  _ISalpha = ((2) < 8 ? ((1 << (2)) << 8) : ((1 << (2)) >> 8)),
  _ISdigit = ((3) < 8 ? ((1 << (3)) << 8) : ((1 << (3)) >> 8)),
  _ISxdigit = ((4) < 8 ? ((1 << (4)) << 8) : ((1 << (4)) >> 8)),
  _ISspace = ((5) < 8 ? ((1 << (5)) << 8) : ((1 << (5)) >> 8)),
  _ISprint = ((6) < 8 ? ((1 << (6)) << 8) : ((1 << (6)) >> 8)),
  _ISgraph = ((7) < 8 ? ((1 << (7)) << 8) : ((1 << (7)) >> 8)),
  _ISblank = ((8) < 8 ? ((1 << (8)) << 8) : ((1 << (8)) >> 8)),
  _IScntrl = ((9) < 8 ? ((1 << (9)) << 8) : ((1 << (9)) >> 8)),
  _ISpunct = ((10) < 8 ? ((1 << (10)) << 8) : ((1 << (10)) >> 8)),
  _ISalnum = ((11) < 8 ? ((1 << (11)) << 8) : ((1 << (11)) >> 8))
};
# 81 "/usr/include/ctype.h" 3
extern __const unsigned short int **__ctype_b_loc (void)
     __attribute__ ((__nothrow__)) __attribute__ ((__const));
extern __const __int32_t **__ctype_tolower_loc (void)
     __attribute__ ((__nothrow__)) __attribute__ ((__const));
extern __const __int32_t **__ctype_toupper_loc (void)
     __attribute__ ((__nothrow__)) __attribute__ ((__const));
# 96 "/usr/include/ctype.h" 3






extern int isalnum (int) __attribute__ ((__nothrow__));
extern int isalpha (int) __attribute__ ((__nothrow__));
extern int iscntrl (int) __attribute__ ((__nothrow__));
extern int isdigit (int) __attribute__ ((__nothrow__));
extern int islower (int) __attribute__ ((__nothrow__));
extern int isgraph (int) __attribute__ ((__nothrow__));
extern int isprint (int) __attribute__ ((__nothrow__));
extern int ispunct (int) __attribute__ ((__nothrow__));
extern int isspace (int) __attribute__ ((__nothrow__));
extern int isupper (int) __attribute__ ((__nothrow__));
extern int isxdigit (int) __attribute__ ((__nothrow__));



extern int tolower (int __c) __attribute__ ((__nothrow__));


extern int toupper (int __c) __attribute__ ((__nothrow__));


# 142 "/usr/include/ctype.h" 3
extern int isascii (int __c) __attribute__ ((__nothrow__));



extern int toascii (int __c) __attribute__ ((__nothrow__));



extern int _toupper (int) __attribute__ ((__nothrow__));
extern int _tolower (int) __attribute__ ((__nothrow__));
# 323 "/usr/include/ctype.h" 3

# 127 "exStbDemo.c" 2
# 1 "/usr/include/termios.h" 1 3
# 36 "/usr/include/termios.h" 3




# 1 "/usr/include/bits/termios.h" 1 3
# 25 "/usr/include/bits/termios.h" 3
typedef unsigned char cc_t;
typedef unsigned int speed_t;
typedef unsigned int tcflag_t;


struct termios
  {
    tcflag_t c_iflag;
    tcflag_t c_oflag;
    tcflag_t c_cflag;
    tcflag_t c_lflag;
    cc_t c_line;
    cc_t c_cc[32];
    speed_t c_ispeed;
    speed_t c_ospeed;


  };
# 41 "/usr/include/termios.h" 2 3
# 49 "/usr/include/termios.h" 3
extern speed_t cfgetospeed (__const struct termios *__termios_p) __attribute__ ((__nothrow__));


extern speed_t cfgetispeed (__const struct termios *__termios_p) __attribute__ ((__nothrow__));


extern int cfsetospeed (struct termios *__termios_p, speed_t __speed) __attribute__ ((__nothrow__));


extern int cfsetispeed (struct termios *__termios_p, speed_t __speed) __attribute__ ((__nothrow__));



extern int cfsetspeed (struct termios *__termios_p, speed_t __speed) __attribute__ ((__nothrow__));




extern int tcgetattr (int __fd, struct termios *__termios_p) __attribute__ ((__nothrow__));



extern int tcsetattr (int __fd, int __optional_actions,
        __const struct termios *__termios_p) __attribute__ ((__nothrow__));




extern void cfmakeraw (struct termios *__termios_p) __attribute__ ((__nothrow__));



extern int tcsendbreak (int __fd, int __duration) __attribute__ ((__nothrow__));





extern int tcdrain (int __fd);



extern int tcflush (int __fd, int __queue_selector) __attribute__ ((__nothrow__));



extern int tcflow (int __fd, int __action) __attribute__ ((__nothrow__));
# 105 "/usr/include/termios.h" 3
# 1 "/usr/include/sys/ttydefaults.h" 1 3
# 106 "/usr/include/termios.h" 2 3



# 128 "exStbDemo.c" 2
# 1 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h" 1
# 81 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
# 1 "/home/lucas/software/pr11/stb225/sde2/inc/tmCompId.h" 1
# 103 "/home/lucas/software/pr11/stb225/sde2/inc/tmCompId.h"
# 1 "/home/lucas/software/pr11/stb225/sde2/inc/tmtypes.h" 1
# 39 "/home/lucas/software/pr11/stb225/sde2/inc/tmtypes.h"
# 1 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h" 1
# 78 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
# 1 "/home/lucas/software/pr11/stb225/build_ctpim/sde2/comps/generated/lib/mipsgnu_linux_el_4KEc/tmFlags.h" 1
# 79 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h" 2
# 130 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
# 1 "/usr/include/stdint.h" 1 3
# 27 "/usr/include/stdint.h" 3
# 1 "/usr/include/bits/wchar.h" 1 3
# 28 "/usr/include/stdint.h" 2 3
# 1 "/usr/include/bits/wordsize.h" 1 3
# 29 "/usr/include/stdint.h" 2 3
# 49 "/usr/include/stdint.h" 3
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;

typedef unsigned int uint32_t;





__extension__
typedef unsigned long long int uint64_t;






typedef signed char int_least8_t;
typedef short int int_least16_t;
typedef int int_least32_t;



__extension__
typedef long long int int_least64_t;



typedef unsigned char uint_least8_t;
typedef unsigned short int uint_least16_t;
typedef unsigned int uint_least32_t;



__extension__
typedef unsigned long long int uint_least64_t;






typedef signed char int_fast8_t;





typedef int int_fast16_t;
typedef int int_fast32_t;
__extension__
typedef long long int int_fast64_t;



typedef unsigned char uint_fast8_t;





typedef unsigned int uint_fast16_t;
typedef unsigned int uint_fast32_t;
__extension__
typedef unsigned long long int uint_fast64_t;
# 129 "/usr/include/stdint.h" 3
typedef unsigned int uintptr_t;
# 138 "/usr/include/stdint.h" 3
__extension__
typedef long long int intmax_t;
__extension__
typedef unsigned long long int uintmax_t;
# 131 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h" 2







    typedef _Bool bool;
# 149 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
  typedef int8_t Int8;
  typedef int16_t Int16;
  typedef int32_t Int32;
  typedef uint8_t UInt8;
  typedef uint16_t UInt16;
  typedef uint32_t UInt32;






  typedef uint8_t Bool;
# 203 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef char char_t;
# 222 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef float Float;
typedef char Char;
typedef int Int;
typedef unsigned int UInt;
typedef char *String;





typedef char *Address;
typedef char const *ConstAddress;
typedef unsigned char Byte;
typedef float Float32;
typedef double Float64;
typedef void *Pointer;
typedef void const *ConstPointer;
typedef char const *ConstString;

typedef Int Endian;





typedef enum { TM32 = 0, TM32V2, TM64=100 } TMArch;
extern char* TMArch_names[];

typedef struct tmVersion
{
    UInt8 majorVersion;
    UInt8 minorVersion;
    UInt16 buildVersion;
} tmVersion_t, *ptmVersion_t;
# 267 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef signed int IBits32;
typedef unsigned int UBits32;

typedef Int8 *pInt8;
typedef Int16 *pInt16;
typedef Int32 *pInt32;
typedef IBits32 *pIBits32;
typedef UBits32 *pUBits32;
typedef UInt8 *pUInt8;
typedef UInt16 *pUInt16;
typedef UInt32 *pUInt32;
typedef void Void, *pVoid;
typedef Float *pFloat;
typedef double Double, *pDouble;
typedef Bool *pBool;
typedef Char *pChar;
typedef Int *pInt;
typedef UInt *pUInt;
typedef String *pString;
# 350 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef signed long long Int64, *pInt64;
typedef unsigned long long UInt64, *pUInt64;
# 395 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef UInt32 tmErrorCode_t;
typedef UInt32 tmProgressCode_t;


typedef UInt64 tmTimeStamp_t, *ptmTimeStamp_t;





typedef union tmColor3
{
    UBits32 u32;
# 428 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
    struct {
        UBits32 blue : 8;
        UBits32 green : 8;
        UBits32 red : 8;
        UBits32 : 8;
    } rgb;
    struct {
        UBits32 v : 8;
        UBits32 u : 8;
        UBits32 y : 8;
        UBits32 : 8;
    } yuv;
    struct {
        UBits32 l : 8;
        UBits32 m : 8;
        UBits32 u : 8;
        UBits32 : 8;
    } uml;

} tmColor3_t, *ptmColor3_t;

typedef union tmColor4
{
    UBits32 u32;
# 472 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
    struct {
        UBits32 blue : 8;
        UBits32 green : 8;
        UBits32 red : 8;
        UBits32 alpha : 8;
    } argb;
    struct {
        UBits32 v : 8;
        UBits32 u : 8;
        UBits32 y : 8;
        UBits32 alpha : 8;
    } ayuv;
    struct {
        UBits32 l : 8;
        UBits32 m : 8;
        UBits32 u : 8;
        UBits32 alpha : 8;
    } auml;

} tmColor4_t, *ptmColor4_t;




typedef enum tmPowerState
{
    tmPowerOn,
    tmPowerStandby,
    tmPowerSuspend,
    tmPowerOff

} tmPowerState_t, *ptmPowerState_t;




typedef struct tmSWVersion
{
    UInt32 compatibilityNr;
    UInt32 majorVersionNr;
    UInt32 minorVersionNr;

} tmSWVersion_t, *ptmSWVersion_t;
# 540 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef Int tmUnitSelect_t, *ptmUnitSelect_t;
# 570 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef Int tmInstance_t, *ptmInstance_t;


typedef Void (*ptmCallback_t) (UInt32 events, Void *pData, UInt32 userData);
# 39 "/home/lucas/software/pr11/stb225/sde2/inc/tmtypes.h" 2
# 104 "/home/lucas/software/pr11/stb225/sde2/inc/tmCompId.h" 2
# 82 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h" 2
# 1 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h" 1
# 83 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h" 2
# 100 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
typedef struct
{
    uint32_t size;
    uint32_t alignment;
    uint32_t address;
}tmdlAudioIO_Buffer_t;


typedef enum
{
    tmdlAudioIO_Output_I2S_0,
    tmdlAudioIO_Output_I2S_1,
    tmdlAudioIO_Output_I2S_2,
    tmdlAudioIO_Output_I2S_3,
    tmdlAudioIO_Output_I2S_4,
    tmdlAudioIO_Output_SPDIF,
    tmdlAudioIO_Output_Guard
} tmdlAudioIO_Output_t;


typedef enum
{
    tmdlAudioIO_SampleRate32KHZ,
    tmdlAudioIO_SampleRate44KHZ,
    tmdlAudioIO_SampleRate48KHZ,
    tmdlAudioIO_SampleRate96KHZ,
    tmdlAudioIO_SampleRateGuard,
} tmdlAudioIO_SampleRate_t;


typedef struct
{
    tmdlAudioIO_Buffer_t buffer[tmdlAudioIO_Output_Guard];

}tmdlAudioIO_InstanceSetup_t;

typedef struct
{
    tmdlAudioIO_Buffer_t buffer[tmdlAudioIO_Output_Guard];
}tmdlAudioIO_PtsBufferIDs_t;

typedef enum
{
    tmdlAudioIO_AcpClk_Internal = 0,
    tmdlAudioIO_AcpClk_External
}tmdlAudioIO_AcpClkSource_t;

typedef enum
{
    tmdlAudioIO_SyncModePTSLocked = 0,
    tmdlAudioIO_SyncModeFreeRunning,
    tmdlAudioIO_SyncModeBufControlled,
    tmdlAudioIO_SyncModeSwControlled
}tmdlAudioIO_SyncMode_t ;


typedef void (*pCallbackFunc_t)(int context, int event, void * pdata);
# 195 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
tmErrorCode_t tmdlAudioIO_Open(
    tmInstance_t *pAioInst
    );
# 208 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
tmErrorCode_t tmdlAudioIO_OpenM(
    tmInstance_t *pAioInst,
    tmUnitSelect_t unitSelect
    );
# 222 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
tmErrorCode_t tmdlAudioIO_Close(
    tmInstance_t aioInst
    );
# 233 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
tmErrorCode_t tmdlAudioIO_GetInstanceSetup(
    tmInstance_t AioInst,
    tmdlAudioIO_InstanceSetup_t ** ppSetup
    );
# 245 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
tmErrorCode_t tmdlAudioIO_InstanceSetup(
    tmInstance_t AioInst,
    tmdlAudioIO_InstanceSetup_t * pSetup
    );
# 261 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIO_FillBuffer(
    tmdlAudioIO_Output_t output,
    char *data_ptr,
    UInt32 data_size,
    UInt32 *data_loaded_ptr
    );
# 286 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIOStart(
    tmInstance_t kAudioIoInstance,
    tmdlAudioIO_Output_t kOutput
    );
# 305 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIOStop(
    tmInstance_t kAudioIoInstance,
    tmdlAudioIO_Output_t kOutput
    );
# 325 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIOPause(
    tmInstance_t kAudioIOInstance,
    tmdlAudioIO_Output_t kOutput,
    bool kPause
    );
# 343 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIO_Flush(
    tmdlAudioIO_Output_t output,
    Int32 *nb_bytes
    );
# 359 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIO_SetSampleRate(
    tmInstance_t aioInst,
    tmdlAudioIO_SampleRate_t sampleRate
    );
# 373 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIO_SetAcpClocks(
    tmInstance_t instance,
    tmdlAudioIO_AcpClkSource_t acp_clk
);
# 386 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIO_SetStc(
    tmInstance_t instance,
    tmTimeStamp_t stcValue
);
# 405 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIO_GetStc(
    tmInstance_t kAudioIoInstance,
    tmTimeStamp_t *pkStcValue
    );
# 418 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIO_SetSyncMode(
    tmInstance_t instance,
    tmdlAudioIO_Output_t output,
    tmdlAudioIO_SyncMode_t syncMode
    );
# 434 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIO_GetSpdifSource(
    tmInstance_t instance,
    tmdlAudioIO_Output_t *pSource
    );
# 449 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIO_SetSpdifSource(
    tmInstance_t instance,
    tmdlAudioIO_Output_t source
    );
# 469 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern void tmdlAudioIORegisterCallback(
    tmInstance_t kAudioIOInstance,
    tmdlAudioIO_Output_t kOutput,
    uint32_t context,
    pCallbackFunc_t pCallback
    );
# 484 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIO_AudioBeep(
    tmInstance_t aioInst,
    uint32_t VolAttnLevel
    );
# 499 "/home/lucas/software/pr11/stb225/src/cpiDrivers/intfs/ItmdlAudioIO/inc/tmdlAudioIO.h"
extern tmErrorCode_t tmdlAudioIO_SelectBeep(
    tmInstance_t aioInst,
    uint32_t beepSel
    );
# 129 "exStbDemo.c" 2

# 1 "/home/lucas/software/pr11/stb225/src/comps/phStbFB/inc/phStbFB.h" 1
# 40 "/home/lucas/software/pr11/stb225/src/comps/phStbFB/inc/phStbFB.h"
# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/ioctl.h" 1



# 1 "/usr/include/asm/ioctl.h" 1 3
# 5 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/ioctl.h" 2
# 41 "/home/lucas/software/pr11/stb225/src/comps/phStbFB/inc/phStbFB.h" 2
# 131 "exStbDemo.c" 2

# 1 "app_info.h" 1
# 110 "app_info.h"
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stdbool.h" 1 3 4
# 111 "app_info.h" 2

# 1 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/dvb/frontend.h" 1
# 32 "/home/lucas/software/pr11/stb225_tarballs/linux-2.6.24_nxp/include/linux/dvb/frontend.h"
typedef enum fe_type {
 FE_QPSK,
 FE_QAM,
 FE_OFDM,
 FE_ATSC
} fe_type_t;


typedef enum fe_caps {
 FE_IS_STUPID = 0,
 FE_CAN_INVERSION_AUTO = 0x1,
 FE_CAN_FEC_1_2 = 0x2,
 FE_CAN_FEC_2_3 = 0x4,
 FE_CAN_FEC_3_4 = 0x8,
 FE_CAN_FEC_4_5 = 0x10,
 FE_CAN_FEC_5_6 = 0x20,
 FE_CAN_FEC_6_7 = 0x40,
 FE_CAN_FEC_7_8 = 0x80,
 FE_CAN_FEC_8_9 = 0x100,
 FE_CAN_FEC_AUTO = 0x200,
 FE_CAN_QPSK = 0x400,
 FE_CAN_QAM_16 = 0x800,
 FE_CAN_QAM_32 = 0x1000,
 FE_CAN_QAM_64 = 0x2000,
 FE_CAN_QAM_128 = 0x4000,
 FE_CAN_QAM_256 = 0x8000,
 FE_CAN_QAM_AUTO = 0x10000,
 FE_CAN_TRANSMISSION_MODE_AUTO = 0x20000,
 FE_CAN_BANDWIDTH_AUTO = 0x40000,
 FE_CAN_GUARD_INTERVAL_AUTO = 0x80000,
 FE_CAN_HIERARCHY_AUTO = 0x100000,
 FE_CAN_8VSB = 0x200000,
 FE_CAN_16VSB = 0x400000,
 FE_NEEDS_BENDING = 0x20000000,
 FE_CAN_RECOVER = 0x40000000,
 FE_CAN_MUTE_TS = 0x80000000
} fe_caps_t;


struct dvb_frontend_info {
 char name[128];
 fe_type_t type;
 __u32 frequency_min;
 __u32 frequency_max;
 __u32 frequency_stepsize;
 __u32 frequency_tolerance;
 __u32 symbol_rate_min;
 __u32 symbol_rate_max;
 __u32 symbol_rate_tolerance;
 __u32 notifier_delay;
 fe_caps_t caps;
};






struct dvb_diseqc_master_cmd {
 __u8 msg [6];
 __u8 msg_len;
};


struct dvb_diseqc_slave_reply {
 __u8 msg [4];
 __u8 msg_len;
 int timeout;
};


typedef enum fe_sec_voltage {
 SEC_VOLTAGE_13,
 SEC_VOLTAGE_18,
 SEC_VOLTAGE_OFF
} fe_sec_voltage_t;


typedef enum fe_sec_tone_mode {
 SEC_TONE_ON,
 SEC_TONE_OFF
} fe_sec_tone_mode_t;


typedef enum fe_sec_mini_cmd {
 SEC_MINI_A,
 SEC_MINI_B
} fe_sec_mini_cmd_t;


typedef enum fe_status {
 FE_HAS_SIGNAL = 0x01,
 FE_HAS_CARRIER = 0x02,
 FE_HAS_VITERBI = 0x04,
 FE_HAS_SYNC = 0x08,
 FE_HAS_LOCK = 0x10,
 FE_TIMEDOUT = 0x20,
 FE_REINIT = 0x40
} fe_status_t;


typedef enum fe_spectral_inversion {
 INVERSION_OFF,
 INVERSION_ON,
 INVERSION_AUTO
} fe_spectral_inversion_t;


typedef enum fe_code_rate {
 FEC_NONE = 0,
 FEC_1_2,
 FEC_2_3,
 FEC_3_4,
 FEC_4_5,
 FEC_5_6,
 FEC_6_7,
 FEC_7_8,
 FEC_8_9,
 FEC_AUTO
} fe_code_rate_t;


typedef enum fe_modulation {
 QPSK,
 QAM_16,
 QAM_32,
 QAM_64,
 QAM_128,
 QAM_256,
 QAM_AUTO,
 VSB_8,
 VSB_16
} fe_modulation_t;

typedef enum fe_transmit_mode {
 TRANSMISSION_MODE_2K,
 TRANSMISSION_MODE_8K,
 TRANSMISSION_MODE_AUTO
} fe_transmit_mode_t;

typedef enum fe_bandwidth {
 BANDWIDTH_8_MHZ,
 BANDWIDTH_7_MHZ,
 BANDWIDTH_6_MHZ,
 BANDWIDTH_AUTO
} fe_bandwidth_t;


typedef enum fe_guard_interval {
 GUARD_INTERVAL_1_32,
 GUARD_INTERVAL_1_16,
 GUARD_INTERVAL_1_8,
 GUARD_INTERVAL_1_4,
 GUARD_INTERVAL_AUTO
} fe_guard_interval_t;


typedef enum fe_hierarchy {
 HIERARCHY_NONE,
 HIERARCHY_1,
 HIERARCHY_2,
 HIERARCHY_4,
 HIERARCHY_AUTO
} fe_hierarchy_t;


struct dvb_qpsk_parameters {
 __u32 symbol_rate;
 fe_code_rate_t fec_inner;
};

struct dvb_qam_parameters {
 __u32 symbol_rate;
 fe_code_rate_t fec_inner;
 fe_modulation_t modulation;
};

struct dvb_vsb_parameters {
 fe_modulation_t modulation;
};

struct dvb_ofdm_parameters {
 fe_bandwidth_t bandwidth;
 fe_code_rate_t code_rate_HP;
 fe_code_rate_t code_rate_LP;
 fe_modulation_t constellation;
 fe_transmit_mode_t transmission_mode;
 fe_guard_interval_t guard_interval;
 fe_hierarchy_t hierarchy_information;
};


struct dvb_frontend_parameters {
 __u32 frequency;

 fe_spectral_inversion_t inversion;
 union {
  struct dvb_qpsk_parameters qpsk;
  struct dvb_qam_parameters qam;
  struct dvb_ofdm_parameters ofdm;
  struct dvb_vsb_parameters vsb;
 } u;
};


struct dvb_frontend_event {
 fe_status_t status;
 struct dvb_frontend_parameters parameters;
};
# 113 "app_info.h" 2
# 1 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h" 1
# 32 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
# 1 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/dfb_types.h" 1
# 51 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/dfb_types.h"
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
# 33 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h" 2


# 1 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb_keyboard.h" 1
# 41 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb_keyboard.h"
typedef enum {
     DIKT_UNICODE = 0x0000,

     DIKT_SPECIAL = 0xF000,
     DIKT_FUNCTION = 0xF100,
     DIKT_MODIFIER = 0xF200,
     DIKT_LOCK = 0xF300,
     DIKT_DEAD = 0xF400,
     DIKT_CUSTOM = 0xF500,
     DIKT_IDENTIFIER = 0xF600
} DFBInputDeviceKeyType;
# 72 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb_keyboard.h"
typedef enum {
     DIMKI_SHIFT,
     DIMKI_CONTROL,
     DIMKI_ALT,
     DIMKI_ALTGR,
     DIMKI_META,
     DIMKI_SUPER,
     DIMKI_HYPER,

     DIMKI_FIRST = DIMKI_SHIFT,
     DIMKI_LAST = DIMKI_HYPER
} DFBInputDeviceModifierKeyIdentifier;




typedef enum {
     DIKI_UNKNOWN = ((DIKT_IDENTIFIER) | (0)),

     DIKI_A,
     DIKI_B,
     DIKI_C,
     DIKI_D,
     DIKI_E,
     DIKI_F,
     DIKI_G,
     DIKI_H,
     DIKI_I,
     DIKI_J,
     DIKI_K,
     DIKI_L,
     DIKI_M,
     DIKI_N,
     DIKI_O,
     DIKI_P,
     DIKI_Q,
     DIKI_R,
     DIKI_S,
     DIKI_T,
     DIKI_U,
     DIKI_V,
     DIKI_W,
     DIKI_X,
     DIKI_Y,
     DIKI_Z,

     DIKI_0,
     DIKI_1,
     DIKI_2,
     DIKI_3,
     DIKI_4,
     DIKI_5,
     DIKI_6,
     DIKI_7,
     DIKI_8,
     DIKI_9,

     DIKI_F1,
     DIKI_F2,
     DIKI_F3,
     DIKI_F4,
     DIKI_F5,
     DIKI_F6,
     DIKI_F7,
     DIKI_F8,
     DIKI_F9,
     DIKI_F10,
     DIKI_F11,
     DIKI_F12,

     DIKI_SHIFT_L,
     DIKI_SHIFT_R,
     DIKI_CONTROL_L,
     DIKI_CONTROL_R,
     DIKI_ALT_L,
     DIKI_ALT_R,
     DIKI_META_L,
     DIKI_META_R,
     DIKI_SUPER_L,
     DIKI_SUPER_R,
     DIKI_HYPER_L,
     DIKI_HYPER_R,

     DIKI_CAPS_LOCK,
     DIKI_NUM_LOCK,
     DIKI_SCROLL_LOCK,

     DIKI_ESCAPE,
     DIKI_LEFT,
     DIKI_RIGHT,
     DIKI_UP,
     DIKI_DOWN,
     DIKI_TAB,
     DIKI_ENTER,
     DIKI_SPACE,
     DIKI_BACKSPACE,
     DIKI_INSERT,
     DIKI_DELETE,
     DIKI_HOME,
     DIKI_END,
     DIKI_PAGE_UP,
     DIKI_PAGE_DOWN,
     DIKI_PRINT,
     DIKI_PAUSE,





     DIKI_QUOTE_LEFT,
     DIKI_MINUS_SIGN,
     DIKI_EQUALS_SIGN,
     DIKI_BRACKET_LEFT,
     DIKI_BRACKET_RIGHT,
     DIKI_BACKSLASH,
     DIKI_SEMICOLON,
     DIKI_QUOTE_RIGHT,
     DIKI_COMMA,
     DIKI_PERIOD,
     DIKI_SLASH,

     DIKI_LESS_SIGN,

     DIKI_KP_DIV,
     DIKI_KP_MULT,
     DIKI_KP_MINUS,
     DIKI_KP_PLUS,
     DIKI_KP_ENTER,
     DIKI_KP_SPACE,
     DIKI_KP_TAB,
     DIKI_KP_F1,
     DIKI_KP_F2,
     DIKI_KP_F3,
     DIKI_KP_F4,
     DIKI_KP_EQUAL,
     DIKI_KP_SEPARATOR,

     DIKI_KP_DECIMAL,
     DIKI_KP_0,
     DIKI_KP_1,
     DIKI_KP_2,
     DIKI_KP_3,
     DIKI_KP_4,
     DIKI_KP_5,
     DIKI_KP_6,
     DIKI_KP_7,
     DIKI_KP_8,
     DIKI_KP_9,

     DIKI_KEYDEF_END,
     DIKI_NUMBER_OF_KEYS = DIKI_KEYDEF_END - ((DIKT_IDENTIFIER) | (0))

} DFBInputDeviceKeyIdentifier;




typedef enum {






     DIKS_NULL = ((DIKT_UNICODE) | (0x00)),
     DIKS_BACKSPACE = ((DIKT_UNICODE) | (0x08)),
     DIKS_TAB = ((DIKT_UNICODE) | (0x09)),
     DIKS_RETURN = ((DIKT_UNICODE) | (0x0D)),
     DIKS_CANCEL = ((DIKT_UNICODE) | (0x18)),
     DIKS_ESCAPE = ((DIKT_UNICODE) | (0x1B)),
     DIKS_SPACE = ((DIKT_UNICODE) | (0x20)),
     DIKS_EXCLAMATION_MARK = ((DIKT_UNICODE) | (0x21)),
     DIKS_QUOTATION = ((DIKT_UNICODE) | (0x22)),
     DIKS_NUMBER_SIGN = ((DIKT_UNICODE) | (0x23)),
     DIKS_DOLLAR_SIGN = ((DIKT_UNICODE) | (0x24)),
     DIKS_PERCENT_SIGN = ((DIKT_UNICODE) | (0x25)),
     DIKS_AMPERSAND = ((DIKT_UNICODE) | (0x26)),
     DIKS_APOSTROPHE = ((DIKT_UNICODE) | (0x27)),
     DIKS_PARENTHESIS_LEFT = ((DIKT_UNICODE) | (0x28)),
     DIKS_PARENTHESIS_RIGHT = ((DIKT_UNICODE) | (0x29)),
     DIKS_ASTERISK = ((DIKT_UNICODE) | (0x2A)),
     DIKS_PLUS_SIGN = ((DIKT_UNICODE) | (0x2B)),
     DIKS_COMMA = ((DIKT_UNICODE) | (0x2C)),
     DIKS_MINUS_SIGN = ((DIKT_UNICODE) | (0x2D)),
     DIKS_PERIOD = ((DIKT_UNICODE) | (0x2E)),
     DIKS_SLASH = ((DIKT_UNICODE) | (0x2F)),
     DIKS_0 = ((DIKT_UNICODE) | (0x30)),
     DIKS_1 = ((DIKT_UNICODE) | (0x31)),
     DIKS_2 = ((DIKT_UNICODE) | (0x32)),
     DIKS_3 = ((DIKT_UNICODE) | (0x33)),
     DIKS_4 = ((DIKT_UNICODE) | (0x34)),
     DIKS_5 = ((DIKT_UNICODE) | (0x35)),
     DIKS_6 = ((DIKT_UNICODE) | (0x36)),
     DIKS_7 = ((DIKT_UNICODE) | (0x37)),
     DIKS_8 = ((DIKT_UNICODE) | (0x38)),
     DIKS_9 = ((DIKT_UNICODE) | (0x39)),
     DIKS_COLON = ((DIKT_UNICODE) | (0x3A)),
     DIKS_SEMICOLON = ((DIKT_UNICODE) | (0x3B)),
     DIKS_LESS_THAN_SIGN = ((DIKT_UNICODE) | (0x3C)),
     DIKS_EQUALS_SIGN = ((DIKT_UNICODE) | (0x3D)),
     DIKS_GREATER_THAN_SIGN = ((DIKT_UNICODE) | (0x3E)),
     DIKS_QUESTION_MARK = ((DIKT_UNICODE) | (0x3F)),
     DIKS_AT = ((DIKT_UNICODE) | (0x40)),
     DIKS_CAPITAL_A = ((DIKT_UNICODE) | (0x41)),
     DIKS_CAPITAL_B = ((DIKT_UNICODE) | (0x42)),
     DIKS_CAPITAL_C = ((DIKT_UNICODE) | (0x43)),
     DIKS_CAPITAL_D = ((DIKT_UNICODE) | (0x44)),
     DIKS_CAPITAL_E = ((DIKT_UNICODE) | (0x45)),
     DIKS_CAPITAL_F = ((DIKT_UNICODE) | (0x46)),
     DIKS_CAPITAL_G = ((DIKT_UNICODE) | (0x47)),
     DIKS_CAPITAL_H = ((DIKT_UNICODE) | (0x48)),
     DIKS_CAPITAL_I = ((DIKT_UNICODE) | (0x49)),
     DIKS_CAPITAL_J = ((DIKT_UNICODE) | (0x4A)),
     DIKS_CAPITAL_K = ((DIKT_UNICODE) | (0x4B)),
     DIKS_CAPITAL_L = ((DIKT_UNICODE) | (0x4C)),
     DIKS_CAPITAL_M = ((DIKT_UNICODE) | (0x4D)),
     DIKS_CAPITAL_N = ((DIKT_UNICODE) | (0x4E)),
     DIKS_CAPITAL_O = ((DIKT_UNICODE) | (0x4F)),
     DIKS_CAPITAL_P = ((DIKT_UNICODE) | (0x50)),
     DIKS_CAPITAL_Q = ((DIKT_UNICODE) | (0x51)),
     DIKS_CAPITAL_R = ((DIKT_UNICODE) | (0x52)),
     DIKS_CAPITAL_S = ((DIKT_UNICODE) | (0x53)),
     DIKS_CAPITAL_T = ((DIKT_UNICODE) | (0x54)),
     DIKS_CAPITAL_U = ((DIKT_UNICODE) | (0x55)),
     DIKS_CAPITAL_V = ((DIKT_UNICODE) | (0x56)),
     DIKS_CAPITAL_W = ((DIKT_UNICODE) | (0x57)),
     DIKS_CAPITAL_X = ((DIKT_UNICODE) | (0x58)),
     DIKS_CAPITAL_Y = ((DIKT_UNICODE) | (0x59)),
     DIKS_CAPITAL_Z = ((DIKT_UNICODE) | (0x5A)),
     DIKS_SQUARE_BRACKET_LEFT = ((DIKT_UNICODE) | (0x5B)),
     DIKS_BACKSLASH = ((DIKT_UNICODE) | (0x5C)),
     DIKS_SQUARE_BRACKET_RIGHT = ((DIKT_UNICODE) | (0x5D)),
     DIKS_CIRCUMFLEX_ACCENT = ((DIKT_UNICODE) | (0x5E)),
     DIKS_UNDERSCORE = ((DIKT_UNICODE) | (0x5F)),
     DIKS_GRAVE_ACCENT = ((DIKT_UNICODE) | (0x60)),
     DIKS_SMALL_A = ((DIKT_UNICODE) | (0x61)),
     DIKS_SMALL_B = ((DIKT_UNICODE) | (0x62)),
     DIKS_SMALL_C = ((DIKT_UNICODE) | (0x63)),
     DIKS_SMALL_D = ((DIKT_UNICODE) | (0x64)),
     DIKS_SMALL_E = ((DIKT_UNICODE) | (0x65)),
     DIKS_SMALL_F = ((DIKT_UNICODE) | (0x66)),
     DIKS_SMALL_G = ((DIKT_UNICODE) | (0x67)),
     DIKS_SMALL_H = ((DIKT_UNICODE) | (0x68)),
     DIKS_SMALL_I = ((DIKT_UNICODE) | (0x69)),
     DIKS_SMALL_J = ((DIKT_UNICODE) | (0x6A)),
     DIKS_SMALL_K = ((DIKT_UNICODE) | (0x6B)),
     DIKS_SMALL_L = ((DIKT_UNICODE) | (0x6C)),
     DIKS_SMALL_M = ((DIKT_UNICODE) | (0x6D)),
     DIKS_SMALL_N = ((DIKT_UNICODE) | (0x6E)),
     DIKS_SMALL_O = ((DIKT_UNICODE) | (0x6F)),
     DIKS_SMALL_P = ((DIKT_UNICODE) | (0x70)),
     DIKS_SMALL_Q = ((DIKT_UNICODE) | (0x71)),
     DIKS_SMALL_R = ((DIKT_UNICODE) | (0x72)),
     DIKS_SMALL_S = ((DIKT_UNICODE) | (0x73)),
     DIKS_SMALL_T = ((DIKT_UNICODE) | (0x74)),
     DIKS_SMALL_U = ((DIKT_UNICODE) | (0x75)),
     DIKS_SMALL_V = ((DIKT_UNICODE) | (0x76)),
     DIKS_SMALL_W = ((DIKT_UNICODE) | (0x77)),
     DIKS_SMALL_X = ((DIKT_UNICODE) | (0x78)),
     DIKS_SMALL_Y = ((DIKT_UNICODE) | (0x79)),
     DIKS_SMALL_Z = ((DIKT_UNICODE) | (0x7A)),
     DIKS_CURLY_BRACKET_LEFT = ((DIKT_UNICODE) | (0x7B)),
     DIKS_VERTICAL_BAR = ((DIKT_UNICODE) | (0x7C)),
     DIKS_CURLY_BRACKET_RIGHT = ((DIKT_UNICODE) | (0x7D)),
     DIKS_TILDE = ((DIKT_UNICODE) | (0x7E)),
     DIKS_DELETE = ((DIKT_UNICODE) | (0x7F)),

     DIKS_ENTER = DIKS_RETURN,




     DIKS_CURSOR_LEFT = ((DIKT_SPECIAL) | (0x00)),
     DIKS_CURSOR_RIGHT = ((DIKT_SPECIAL) | (0x01)),
     DIKS_CURSOR_UP = ((DIKT_SPECIAL) | (0x02)),
     DIKS_CURSOR_DOWN = ((DIKT_SPECIAL) | (0x03)),
     DIKS_INSERT = ((DIKT_SPECIAL) | (0x04)),
     DIKS_HOME = ((DIKT_SPECIAL) | (0x05)),
     DIKS_END = ((DIKT_SPECIAL) | (0x06)),
     DIKS_PAGE_UP = ((DIKT_SPECIAL) | (0x07)),
     DIKS_PAGE_DOWN = ((DIKT_SPECIAL) | (0x08)),
     DIKS_PRINT = ((DIKT_SPECIAL) | (0x09)),
     DIKS_PAUSE = ((DIKT_SPECIAL) | (0x0A)),
     DIKS_OK = ((DIKT_SPECIAL) | (0x0B)),
     DIKS_SELECT = ((DIKT_SPECIAL) | (0x0C)),
     DIKS_GOTO = ((DIKT_SPECIAL) | (0x0D)),
     DIKS_CLEAR = ((DIKT_SPECIAL) | (0x0E)),
     DIKS_POWER = ((DIKT_SPECIAL) | (0x0F)),
     DIKS_POWER2 = ((DIKT_SPECIAL) | (0x10)),
     DIKS_OPTION = ((DIKT_SPECIAL) | (0x11)),
     DIKS_MENU = ((DIKT_SPECIAL) | (0x12)),
     DIKS_HELP = ((DIKT_SPECIAL) | (0x13)),
     DIKS_INFO = ((DIKT_SPECIAL) | (0x14)),
     DIKS_TIME = ((DIKT_SPECIAL) | (0x15)),
     DIKS_VENDOR = ((DIKT_SPECIAL) | (0x16)),

     DIKS_ARCHIVE = ((DIKT_SPECIAL) | (0x17)),
     DIKS_PROGRAM = ((DIKT_SPECIAL) | (0x18)),
     DIKS_CHANNEL = ((DIKT_SPECIAL) | (0x19)),
     DIKS_FAVORITES = ((DIKT_SPECIAL) | (0x1A)),
     DIKS_EPG = ((DIKT_SPECIAL) | (0x1B)),
     DIKS_PVR = ((DIKT_SPECIAL) | (0x1C)),
     DIKS_MHP = ((DIKT_SPECIAL) | (0x1D)),
     DIKS_LANGUAGE = ((DIKT_SPECIAL) | (0x1E)),
     DIKS_TITLE = ((DIKT_SPECIAL) | (0x1F)),
     DIKS_SUBTITLE = ((DIKT_SPECIAL) | (0x20)),
     DIKS_ANGLE = ((DIKT_SPECIAL) | (0x21)),
     DIKS_ZOOM = ((DIKT_SPECIAL) | (0x22)),
     DIKS_MODE = ((DIKT_SPECIAL) | (0x23)),
     DIKS_KEYBOARD = ((DIKT_SPECIAL) | (0x24)),
     DIKS_PC = ((DIKT_SPECIAL) | (0x25)),
     DIKS_SCREEN = ((DIKT_SPECIAL) | (0x26)),

     DIKS_TV = ((DIKT_SPECIAL) | (0x27)),
     DIKS_TV2 = ((DIKT_SPECIAL) | (0x28)),
     DIKS_VCR = ((DIKT_SPECIAL) | (0x29)),
     DIKS_VCR2 = ((DIKT_SPECIAL) | (0x2A)),
     DIKS_SAT = ((DIKT_SPECIAL) | (0x2B)),
     DIKS_SAT2 = ((DIKT_SPECIAL) | (0x2C)),
     DIKS_CD = ((DIKT_SPECIAL) | (0x2D)),
     DIKS_TAPE = ((DIKT_SPECIAL) | (0x2E)),
     DIKS_RADIO = ((DIKT_SPECIAL) | (0x2F)),
     DIKS_TUNER = ((DIKT_SPECIAL) | (0x30)),
     DIKS_PLAYER = ((DIKT_SPECIAL) | (0x31)),
     DIKS_TEXT = ((DIKT_SPECIAL) | (0x32)),
     DIKS_DVD = ((DIKT_SPECIAL) | (0x33)),
     DIKS_AUX = ((DIKT_SPECIAL) | (0x34)),
     DIKS_MP3 = ((DIKT_SPECIAL) | (0x35)),
     DIKS_PHONE = ((DIKT_SPECIAL) | (0x36)),
     DIKS_AUDIO = ((DIKT_SPECIAL) | (0x37)),
     DIKS_VIDEO = ((DIKT_SPECIAL) | (0x38)),

     DIKS_INTERNET = ((DIKT_SPECIAL) | (0x39)),
     DIKS_MAIL = ((DIKT_SPECIAL) | (0x3A)),
     DIKS_NEWS = ((DIKT_SPECIAL) | (0x3B)),
     DIKS_DIRECTORY = ((DIKT_SPECIAL) | (0x3C)),
     DIKS_LIST = ((DIKT_SPECIAL) | (0x3D)),
     DIKS_CALCULATOR = ((DIKT_SPECIAL) | (0x3E)),
     DIKS_MEMO = ((DIKT_SPECIAL) | (0x3F)),
     DIKS_CALENDAR = ((DIKT_SPECIAL) | (0x40)),
     DIKS_EDITOR = ((DIKT_SPECIAL) | (0x41)),

     DIKS_RED = ((DIKT_SPECIAL) | (0x42)),
     DIKS_GREEN = ((DIKT_SPECIAL) | (0x43)),
     DIKS_YELLOW = ((DIKT_SPECIAL) | (0x44)),
     DIKS_BLUE = ((DIKT_SPECIAL) | (0x45)),

     DIKS_CHANNEL_UP = ((DIKT_SPECIAL) | (0x46)),
     DIKS_CHANNEL_DOWN = ((DIKT_SPECIAL) | (0x47)),
     DIKS_BACK = ((DIKT_SPECIAL) | (0x48)),
     DIKS_FORWARD = ((DIKT_SPECIAL) | (0x49)),
     DIKS_FIRST = ((DIKT_SPECIAL) | (0x4A)),
     DIKS_LAST = ((DIKT_SPECIAL) | (0x4B)),
     DIKS_VOLUME_UP = ((DIKT_SPECIAL) | (0x4C)),
     DIKS_VOLUME_DOWN = ((DIKT_SPECIAL) | (0x4D)),
     DIKS_MUTE = ((DIKT_SPECIAL) | (0x4E)),
     DIKS_AB = ((DIKT_SPECIAL) | (0x4F)),
     DIKS_PLAYPAUSE = ((DIKT_SPECIAL) | (0x50)),
     DIKS_PLAY = ((DIKT_SPECIAL) | (0x51)),
     DIKS_STOP = ((DIKT_SPECIAL) | (0x52)),
     DIKS_RESTART = ((DIKT_SPECIAL) | (0x53)),
     DIKS_SLOW = ((DIKT_SPECIAL) | (0x54)),
     DIKS_FAST = ((DIKT_SPECIAL) | (0x55)),
     DIKS_RECORD = ((DIKT_SPECIAL) | (0x56)),
     DIKS_EJECT = ((DIKT_SPECIAL) | (0x57)),
     DIKS_SHUFFLE = ((DIKT_SPECIAL) | (0x58)),
     DIKS_REWIND = ((DIKT_SPECIAL) | (0x59)),
     DIKS_FASTFORWARD = ((DIKT_SPECIAL) | (0x5A)),
     DIKS_PREVIOUS = ((DIKT_SPECIAL) | (0x5B)),
     DIKS_NEXT = ((DIKT_SPECIAL) | (0x5C)),
     DIKS_BEGIN = ((DIKT_SPECIAL) | (0x5D)),

     DIKS_DIGITS = ((DIKT_SPECIAL) | (0x5E)),
     DIKS_TEEN = ((DIKT_SPECIAL) | (0x5F)),
     DIKS_TWEN = ((DIKT_SPECIAL) | (0x60)),

     DIKS_BREAK = ((DIKT_SPECIAL) | (0x61)),
     DIKS_EXIT = ((DIKT_SPECIAL) | (0x62)),
     DIKS_SETUP = ((DIKT_SPECIAL) | (0x63)),

     DIKS_CURSOR_LEFT_UP = ((DIKT_SPECIAL) | (0x64)),
     DIKS_CURSOR_LEFT_DOWN = ((DIKT_SPECIAL) | (0x65)),
     DIKS_CURSOR_UP_RIGHT = ((DIKT_SPECIAL) | (0x66)),
     DIKS_CURSOR_DOWN_RIGHT = ((DIKT_SPECIAL) | (0x67)),






     DIKS_F1 = (((DIKT_FUNCTION) | (1))),
     DIKS_F2 = (((DIKT_FUNCTION) | (2))),
     DIKS_F3 = (((DIKT_FUNCTION) | (3))),
     DIKS_F4 = (((DIKT_FUNCTION) | (4))),
     DIKS_F5 = (((DIKT_FUNCTION) | (5))),
     DIKS_F6 = (((DIKT_FUNCTION) | (6))),
     DIKS_F7 = (((DIKT_FUNCTION) | (7))),
     DIKS_F8 = (((DIKT_FUNCTION) | (8))),
     DIKS_F9 = (((DIKT_FUNCTION) | (9))),
     DIKS_F10 = (((DIKT_FUNCTION) | (10))),
     DIKS_F11 = (((DIKT_FUNCTION) | (11))),
     DIKS_F12 = (((DIKT_FUNCTION) | (12))),




     DIKS_SHIFT = (((DIKT_MODIFIER) | ((1 << DIMKI_SHIFT)))),
     DIKS_CONTROL = (((DIKT_MODIFIER) | ((1 << DIMKI_CONTROL)))),
     DIKS_ALT = (((DIKT_MODIFIER) | ((1 << DIMKI_ALT)))),
     DIKS_ALTGR = (((DIKT_MODIFIER) | ((1 << DIMKI_ALTGR)))),
     DIKS_META = (((DIKT_MODIFIER) | ((1 << DIMKI_META)))),
     DIKS_SUPER = (((DIKT_MODIFIER) | ((1 << DIMKI_SUPER)))),
     DIKS_HYPER = (((DIKT_MODIFIER) | ((1 << DIMKI_HYPER)))),




     DIKS_CAPS_LOCK = ((DIKT_LOCK) | (0x00)),
     DIKS_NUM_LOCK = ((DIKT_LOCK) | (0x01)),
     DIKS_SCROLL_LOCK = ((DIKT_LOCK) | (0x02)),




     DIKS_DEAD_ABOVEDOT = ((DIKT_DEAD) | (0x00)),
     DIKS_DEAD_ABOVERING = ((DIKT_DEAD) | (0x01)),
     DIKS_DEAD_ACUTE = ((DIKT_DEAD) | (0x02)),
     DIKS_DEAD_BREVE = ((DIKT_DEAD) | (0x03)),
     DIKS_DEAD_CARON = ((DIKT_DEAD) | (0x04)),
     DIKS_DEAD_CEDILLA = ((DIKT_DEAD) | (0x05)),
     DIKS_DEAD_CIRCUMFLEX = ((DIKT_DEAD) | (0x06)),
     DIKS_DEAD_DIAERESIS = ((DIKT_DEAD) | (0x07)),
     DIKS_DEAD_DOUBLEACUTE = ((DIKT_DEAD) | (0x08)),
     DIKS_DEAD_GRAVE = ((DIKT_DEAD) | (0x09)),
     DIKS_DEAD_IOTA = ((DIKT_DEAD) | (0x0A)),
     DIKS_DEAD_MACRON = ((DIKT_DEAD) | (0x0B)),
     DIKS_DEAD_OGONEK = ((DIKT_DEAD) | (0x0C)),
     DIKS_DEAD_SEMIVOICED_SOUND = ((DIKT_DEAD) | (0x0D)),
     DIKS_DEAD_TILDE = ((DIKT_DEAD) | (0x0E)),
     DIKS_DEAD_VOICED_SOUND = ((DIKT_DEAD) | (0x0F)),






     DIKS_CUSTOM0 = (((DIKT_CUSTOM) | (0))),
     DIKS_CUSTOM1 = (((DIKT_CUSTOM) | (1))),
     DIKS_CUSTOM2 = (((DIKT_CUSTOM) | (2))),
     DIKS_CUSTOM3 = (((DIKT_CUSTOM) | (3))),
     DIKS_CUSTOM4 = (((DIKT_CUSTOM) | (4))),
     DIKS_CUSTOM5 = (((DIKT_CUSTOM) | (5))),
     DIKS_CUSTOM6 = (((DIKT_CUSTOM) | (6))),
     DIKS_CUSTOM7 = (((DIKT_CUSTOM) | (7))),
     DIKS_CUSTOM8 = (((DIKT_CUSTOM) | (8))),
     DIKS_CUSTOM9 = (((DIKT_CUSTOM) | (9))),
     DIKS_CUSTOM10 = (((DIKT_CUSTOM) | (10))),
     DIKS_CUSTOM11 = (((DIKT_CUSTOM) | (11))),
     DIKS_CUSTOM12 = (((DIKT_CUSTOM) | (12))),
     DIKS_CUSTOM13 = (((DIKT_CUSTOM) | (13))),
     DIKS_CUSTOM14 = (((DIKT_CUSTOM) | (14))),
     DIKS_CUSTOM15 = (((DIKT_CUSTOM) | (15))),
     DIKS_CUSTOM16 = (((DIKT_CUSTOM) | (16))),
     DIKS_CUSTOM17 = (((DIKT_CUSTOM) | (17))),
     DIKS_CUSTOM18 = (((DIKT_CUSTOM) | (18))),
     DIKS_CUSTOM19 = (((DIKT_CUSTOM) | (19))),
     DIKS_CUSTOM20 = (((DIKT_CUSTOM) | (20))),
     DIKS_CUSTOM21 = (((DIKT_CUSTOM) | (21))),
     DIKS_CUSTOM22 = (((DIKT_CUSTOM) | (22))),
     DIKS_CUSTOM23 = (((DIKT_CUSTOM) | (23))),
     DIKS_CUSTOM24 = (((DIKT_CUSTOM) | (24))),
     DIKS_CUSTOM25 = (((DIKT_CUSTOM) | (25))),
     DIKS_CUSTOM26 = (((DIKT_CUSTOM) | (26))),
     DIKS_CUSTOM27 = (((DIKT_CUSTOM) | (27))),
     DIKS_CUSTOM28 = (((DIKT_CUSTOM) | (28))),
     DIKS_CUSTOM29 = (((DIKT_CUSTOM) | (29))),
     DIKS_CUSTOM30 = (((DIKT_CUSTOM) | (30))),
     DIKS_CUSTOM31 = (((DIKT_CUSTOM) | (31))),
     DIKS_CUSTOM32 = (((DIKT_CUSTOM) | (32))),
     DIKS_CUSTOM33 = (((DIKT_CUSTOM) | (33))),
     DIKS_CUSTOM34 = (((DIKT_CUSTOM) | (34))),
     DIKS_CUSTOM35 = (((DIKT_CUSTOM) | (35))),
     DIKS_CUSTOM36 = (((DIKT_CUSTOM) | (36))),
     DIKS_CUSTOM37 = (((DIKT_CUSTOM) | (37))),
     DIKS_CUSTOM38 = (((DIKT_CUSTOM) | (38))),
     DIKS_CUSTOM39 = (((DIKT_CUSTOM) | (39))),
     DIKS_CUSTOM40 = (((DIKT_CUSTOM) | (40))),
     DIKS_CUSTOM41 = (((DIKT_CUSTOM) | (41))),
     DIKS_CUSTOM42 = (((DIKT_CUSTOM) | (42))),
     DIKS_CUSTOM43 = (((DIKT_CUSTOM) | (43))),
     DIKS_CUSTOM44 = (((DIKT_CUSTOM) | (44))),
     DIKS_CUSTOM45 = (((DIKT_CUSTOM) | (45))),
     DIKS_CUSTOM46 = (((DIKT_CUSTOM) | (46))),
     DIKS_CUSTOM47 = (((DIKT_CUSTOM) | (47))),
     DIKS_CUSTOM48 = (((DIKT_CUSTOM) | (48))),
     DIKS_CUSTOM49 = (((DIKT_CUSTOM) | (49))),
     DIKS_CUSTOM50 = (((DIKT_CUSTOM) | (50))),
     DIKS_CUSTOM51 = (((DIKT_CUSTOM) | (51))),
     DIKS_CUSTOM52 = (((DIKT_CUSTOM) | (52))),
     DIKS_CUSTOM53 = (((DIKT_CUSTOM) | (53))),
     DIKS_CUSTOM54 = (((DIKT_CUSTOM) | (54))),
     DIKS_CUSTOM55 = (((DIKT_CUSTOM) | (55))),
     DIKS_CUSTOM56 = (((DIKT_CUSTOM) | (56))),
     DIKS_CUSTOM57 = (((DIKT_CUSTOM) | (57))),
     DIKS_CUSTOM58 = (((DIKT_CUSTOM) | (58))),
     DIKS_CUSTOM59 = (((DIKT_CUSTOM) | (59))),
     DIKS_CUSTOM60 = (((DIKT_CUSTOM) | (60))),
     DIKS_CUSTOM61 = (((DIKT_CUSTOM) | (61))),
     DIKS_CUSTOM62 = (((DIKT_CUSTOM) | (62))),
     DIKS_CUSTOM63 = (((DIKT_CUSTOM) | (63))),
     DIKS_CUSTOM64 = (((DIKT_CUSTOM) | (64))),
     DIKS_CUSTOM65 = (((DIKT_CUSTOM) | (65))),
     DIKS_CUSTOM66 = (((DIKT_CUSTOM) | (66))),
     DIKS_CUSTOM67 = (((DIKT_CUSTOM) | (67))),
     DIKS_CUSTOM68 = (((DIKT_CUSTOM) | (68))),
     DIKS_CUSTOM69 = (((DIKT_CUSTOM) | (69))),
     DIKS_CUSTOM70 = (((DIKT_CUSTOM) | (70))),
     DIKS_CUSTOM71 = (((DIKT_CUSTOM) | (71))),
     DIKS_CUSTOM72 = (((DIKT_CUSTOM) | (72))),
     DIKS_CUSTOM73 = (((DIKT_CUSTOM) | (73))),
     DIKS_CUSTOM74 = (((DIKT_CUSTOM) | (74))),
     DIKS_CUSTOM75 = (((DIKT_CUSTOM) | (75))),
     DIKS_CUSTOM76 = (((DIKT_CUSTOM) | (76))),
     DIKS_CUSTOM77 = (((DIKT_CUSTOM) | (77))),
     DIKS_CUSTOM78 = (((DIKT_CUSTOM) | (78))),
     DIKS_CUSTOM79 = (((DIKT_CUSTOM) | (79))),
     DIKS_CUSTOM80 = (((DIKT_CUSTOM) | (80))),
     DIKS_CUSTOM81 = (((DIKT_CUSTOM) | (81))),
     DIKS_CUSTOM82 = (((DIKT_CUSTOM) | (82))),
     DIKS_CUSTOM83 = (((DIKT_CUSTOM) | (83))),
     DIKS_CUSTOM84 = (((DIKT_CUSTOM) | (84))),
     DIKS_CUSTOM85 = (((DIKT_CUSTOM) | (85))),
     DIKS_CUSTOM86 = (((DIKT_CUSTOM) | (86))),
     DIKS_CUSTOM87 = (((DIKT_CUSTOM) | (87))),
     DIKS_CUSTOM88 = (((DIKT_CUSTOM) | (88))),
     DIKS_CUSTOM89 = (((DIKT_CUSTOM) | (89))),
     DIKS_CUSTOM90 = (((DIKT_CUSTOM) | (90))),
     DIKS_CUSTOM91 = (((DIKT_CUSTOM) | (91))),
     DIKS_CUSTOM92 = (((DIKT_CUSTOM) | (92))),
     DIKS_CUSTOM93 = (((DIKT_CUSTOM) | (93))),
     DIKS_CUSTOM94 = (((DIKT_CUSTOM) | (94))),
     DIKS_CUSTOM95 = (((DIKT_CUSTOM) | (95))),
     DIKS_CUSTOM96 = (((DIKT_CUSTOM) | (96))),
     DIKS_CUSTOM97 = (((DIKT_CUSTOM) | (97))),
     DIKS_CUSTOM98 = (((DIKT_CUSTOM) | (98))),
     DIKS_CUSTOM99 = (((DIKT_CUSTOM) | (99)))
} DFBInputDeviceKeySymbol;




typedef enum {
     DILS_SCROLL = 0x00000001,
     DILS_NUM = 0x00000002,
     DILS_CAPS = 0x00000004
} DFBInputDeviceLockState;




typedef enum {
     DIKSI_BASE = 0x00,

     DIKSI_BASE_SHIFT = 0x01,

     DIKSI_ALT = 0x02,

     DIKSI_ALT_SHIFT = 0x03,


     DIKSI_LAST = DIKSI_ALT_SHIFT
} DFBInputDeviceKeymapSymbolIndex;




typedef struct {
     int code;

     DFBInputDeviceLockState locks;

     DFBInputDeviceKeyIdentifier identifier;
     DFBInputDeviceKeySymbol symbols[DIKSI_LAST+1];

} DFBInputDeviceKeymapEntry;
# 36 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h" 2
# 69 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
extern const unsigned int directfb_major_version;
extern const unsigned int directfb_minor_version;
extern const unsigned int directfb_micro_version;
extern const unsigned int directfb_binary_age;
extern const unsigned int directfb_interface_age;





const char * DirectFBCheckVersion( unsigned int required_major,
                                   unsigned int required_minor,
                                   unsigned int required_micro );





typedef struct _IDirectFB IDirectFB;




typedef struct _IDirectFBScreen IDirectFBScreen;




typedef struct _IDirectFBDisplayLayer IDirectFBDisplayLayer;





typedef struct _IDirectFBSurface IDirectFBSurface;




typedef struct _IDirectFBPalette IDirectFBPalette;





typedef struct _IDirectFBWindow IDirectFBWindow;




typedef struct _IDirectFBInputDevice IDirectFBInputDevice;




typedef struct _IDirectFBEventBuffer IDirectFBEventBuffer;




typedef struct _IDirectFBFont IDirectFBFont;




typedef struct _IDirectFBImageProvider IDirectFBImageProvider;




typedef struct _IDirectFBVideoProvider IDirectFBVideoProvider;




typedef struct _IDirectFBDataBuffer IDirectFBDataBuffer;




typedef struct _IDirectFBGL IDirectFBGL;







typedef enum {
     DFB_OK,
     DFB_FAILURE,
     DFB_INIT,
     DFB_BUG,
     DFB_DEAD,
     DFB_UNSUPPORTED,
     DFB_UNIMPLEMENTED,
     DFB_ACCESSDENIED,
     DFB_INVARG,
     DFB_NOSYSTEMMEMORY,
     DFB_NOVIDEOMEMORY,
     DFB_LOCKED,
     DFB_BUFFEREMPTY,
     DFB_FILENOTFOUND,
     DFB_IO,
     DFB_BUSY,
     DFB_NOIMPL,
     DFB_MISSINGFONT,
     DFB_TIMEOUT,
     DFB_MISSINGIMAGE,
     DFB_THIZNULL,
     DFB_IDNOTFOUND,
     DFB_INVAREA,
     DFB_DESTROYED,
     DFB_FUSION,
     DFB_BUFFERTOOLARGE,
     DFB_INTERRUPTED,
     DFB_NOCONTEXT,
     DFB_TEMPUNAVAIL,
     DFB_LIMITEXCEEDED,
     DFB_NOSUCHMETHOD,
     DFB_NOSUCHINSTANCE,
     DFB_ITEMNOTFOUND,
     DFB_VERSIONMISMATCH,
     DFB_NOSHAREDMEMORY,
     DFB_EOF,
     DFB_SUSPENDED
} DFBResult;




typedef enum {
     DFB_FALSE = 0,
     DFB_TRUE = 1
} DFBBoolean;




typedef struct {
     int x;
     int y;
} DFBPoint;




typedef struct {
     int x;
     int w;
} DFBSpan;




typedef struct {
     int w;
     int h;
} DFBDimension;




typedef struct {
     int x;
     int y;
     int w;
     int h;
} DFBRectangle;






typedef struct {
     float x;
     float y;
     float w;
     float h;
} DFBLocation;






typedef struct {
     int x1;
     int y1;
     int x2;
     int y2;
} DFBRegion;






typedef struct {
     int l;
     int t;
     int r;
     int b;
} DFBInsets;




typedef struct {
     int x1;
     int y1;
     int x2;
     int y2;
     int x3;
     int y3;
} DFBTriangle;




typedef struct {
     u8 a;
     u8 r;
     u8 g;
     u8 b;
} DFBColor;
# 333 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
DFBResult DirectFBError(
                             const char *msg,
                             DFBResult result
                       );




DFBResult DirectFBErrorFatal(
                             const char *msg,
                             DFBResult result
                            );




const char *DirectFBErrorString(
                         DFBResult result
                      );






const char *DirectFBUsageString( void );






DFBResult DirectFBInit(
                         int *argc,
                         char *(*argv[])
                      );






DFBResult DirectFBSetOption(
                         const char *name,
                         const char *value
                      );




DFBResult DirectFBCreate(
                          IDirectFB **interface

                        );


typedef unsigned int DFBScreenID;
typedef unsigned int DFBDisplayLayerID;
typedef unsigned int DFBDisplayLayerSourceID;
typedef unsigned int DFBWindowID;
typedef unsigned int DFBInputDeviceID;
typedef unsigned int DFBTextEncodingID;

typedef u32 DFBDisplayLayerIDs;
# 433 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DFSCL_NORMAL = 0x00000000,




     DFSCL_FULLSCREEN,



     DFSCL_EXCLUSIVE






} DFBCooperativeLevel;




typedef enum {
     DLCAPS_NONE = 0x00000000,

     DLCAPS_SURFACE = 0x00000001,



     DLCAPS_OPACITY = 0x00000002,

     DLCAPS_ALPHACHANNEL = 0x00000004,

     DLCAPS_SCREEN_LOCATION = 0x00000008,


     DLCAPS_FLICKER_FILTERING = 0x00000010,

     DLCAPS_DEINTERLACING = 0x00000020,


     DLCAPS_SRC_COLORKEY = 0x00000040,
     DLCAPS_DST_COLORKEY = 0x00000080,


     DLCAPS_BRIGHTNESS = 0x00000100,
     DLCAPS_CONTRAST = 0x00000200,
     DLCAPS_HUE = 0x00000400,
     DLCAPS_SATURATION = 0x00000800,
     DLCAPS_LEVELS = 0x00001000,

     DLCAPS_FIELD_PARITY = 0x00002000,
     DLCAPS_WINDOWS = 0x00004000,
     DLCAPS_SOURCES = 0x00008000,
     DLCAPS_ALPHA_RAMP = 0x00010000,





     DLCAPS_PREMULTIPLIED = 0x00020000,

     DLCAPS_SCREEN_POSITION = 0x00100000,
     DLCAPS_SCREEN_SIZE = 0x00200000,

     DLCAPS_CLIP_REGIONS = 0x00400000,

     DLCAPS_ALL = 0x0073FFFF
} DFBDisplayLayerCapabilities;




typedef enum {
     DSCCAPS_NONE = 0x00000000,

     DSCCAPS_VSYNC = 0x00000001,

     DSCCAPS_POWER_MANAGEMENT = 0x00000002,

     DSCCAPS_MIXERS = 0x00000010,
     DSCCAPS_ENCODERS = 0x00000020,
     DSCCAPS_OUTPUTS = 0x00000040,

     DSCCAPS_ALL = 0x00000073
} DFBScreenCapabilities;




typedef enum {
     DLOP_NONE = 0x00000000,
     DLOP_ALPHACHANNEL = 0x00000001,


     DLOP_FLICKER_FILTERING = 0x00000002,
     DLOP_DEINTERLACING = 0x00000004,

     DLOP_SRC_COLORKEY = 0x00000008,
     DLOP_DST_COLORKEY = 0x00000010,
     DLOP_OPACITY = 0x00000020,

     DLOP_FIELD_PARITY = 0x00000040
} DFBDisplayLayerOptions;




typedef enum {
     DLBM_UNKNOWN = 0x00000000,

     DLBM_FRONTONLY = 0x00000001,
     DLBM_BACKVIDEO = 0x00000002,
     DLBM_BACKSYSTEM = 0x00000004,
     DLBM_TRIPLE = 0x00000008,
     DLBM_WINDOWS = 0x00000010

} DFBDisplayLayerBufferMode;




typedef enum {
     DSDESC_NONE = 0x00000000,

     DSDESC_CAPS = 0x00000001,
     DSDESC_WIDTH = 0x00000002,
     DSDESC_HEIGHT = 0x00000004,
     DSDESC_PIXELFORMAT = 0x00000008,
     DSDESC_PREALLOCATED = 0x00000010,






     DSDESC_PALETTE = 0x00000020,



     DSDESC_ALL = 0x0000003F
} DFBSurfaceDescriptionFlags;




typedef enum {
     DPDESC_CAPS = 0x00000001,
     DPDESC_SIZE = 0x00000002,
     DPDESC_ENTRIES = 0x00000004


} DFBPaletteDescriptionFlags;




typedef enum {
     DSCAPS_NONE = 0x00000000,

     DSCAPS_PRIMARY = 0x00000001,
     DSCAPS_SYSTEMONLY = 0x00000002,

     DSCAPS_VIDEOONLY = 0x00000004,

     DSCAPS_DOUBLE = 0x00000010,
     DSCAPS_SUBSURFACE = 0x00000020,

     DSCAPS_INTERLACED = 0x00000040,



     DSCAPS_SEPARATED = 0x00000080,



     DSCAPS_STATIC_ALLOC = 0x00000100,





     DSCAPS_TRIPLE = 0x00000200,

     DSCAPS_PREMULTIPLIED = 0x00001000,

     DSCAPS_DEPTH = 0x00010000,

     DSCAPS_ALL = 0x000113F7,


     DSCAPS_FLIPPING = DSCAPS_DOUBLE | DSCAPS_TRIPLE

} DFBSurfaceCapabilities;




typedef enum {
     DPCAPS_NONE = 0x00000000
} DFBPaletteCapabilities;




typedef enum {
     DSDRAW_NOFX = 0x00000000,
     DSDRAW_BLEND = 0x00000001,
     DSDRAW_DST_COLORKEY = 0x00000002,

     DSDRAW_SRC_PREMULTIPLY = 0x00000004,

     DSDRAW_DST_PREMULTIPLY = 0x00000008,
     DSDRAW_DEMULTIPLY = 0x00000010,

     DSDRAW_XOR = 0x00000020

} DFBSurfaceDrawingFlags;




typedef enum {
     DSBLIT_NOFX = 0x00000000,
     DSBLIT_BLEND_ALPHACHANNEL = 0x00000001,

     DSBLIT_BLEND_COLORALPHA = 0x00000002,

     DSBLIT_COLORIZE = 0x00000004,

     DSBLIT_SRC_COLORKEY = 0x00000008,
     DSBLIT_DST_COLORKEY = 0x00000010,

     DSBLIT_SRC_PREMULTIPLY = 0x00000020,

     DSBLIT_DST_PREMULTIPLY = 0x00000040,
     DSBLIT_DEMULTIPLY = 0x00000080,

     DSBLIT_DEINTERLACE = 0x00000100,


     DSBLIT_SRC_PREMULTCOLOR = 0x00000200,
     DSBLIT_XOR = 0x00000400,

     DSBLIT_INDEX_TRANSLATION = 0x00000800,

} DFBSurfaceBlittingFlags;




typedef enum {
     DFXL_NONE = 0x00000000,

     DFXL_FILLRECTANGLE = 0x00000001,
     DFXL_DRAWRECTANGLE = 0x00000002,
     DFXL_DRAWLINE = 0x00000004,
     DFXL_FILLTRIANGLE = 0x00000008,

     DFXL_BLIT = 0x00010000,
     DFXL_STRETCHBLIT = 0x00020000,
     DFXL_TEXTRIANGLES = 0x00040000,

     DFXL_DRAWSTRING = 0x01000000,

     DFXL_ALL = 0x0107000F
} DFBAccelerationMask;
# 716 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DLTF_NONE = 0x00000000,

     DLTF_GRAPHICS = 0x00000001,
     DLTF_VIDEO = 0x00000002,
     DLTF_STILL_PICTURE = 0x00000004,
     DLTF_BACKGROUND = 0x00000008,

     DLTF_ALL = 0x0000000F
} DFBDisplayLayerTypeFlags;





typedef enum {
     DIDTF_NONE = 0x00000000,

     DIDTF_KEYBOARD = 0x00000001,
     DIDTF_MOUSE = 0x00000002,
     DIDTF_JOYSTICK = 0x00000004,
     DIDTF_REMOTE = 0x00000008,
     DIDTF_VIRTUAL = 0x00000010,

     DIDTF_ALL = 0x0000001F
} DFBInputDeviceTypeFlags;




typedef enum {
     DICAPS_KEYS = 0x00000001,
     DICAPS_AXES = 0x00000002,
     DICAPS_BUTTONS = 0x00000004,

     DICAPS_ALL = 0x00000007
} DFBInputDeviceCapabilities;




typedef enum {
     DIBI_LEFT = 0x00000000,
     DIBI_RIGHT = 0x00000001,
     DIBI_MIDDLE = 0x00000002,

     DIBI_FIRST = DIBI_LEFT,

     DIBI_LAST = 0x0000001F
} DFBInputDeviceButtonIdentifier;
# 774 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DIAI_X = 0x00000000,
     DIAI_Y = 0x00000001,
     DIAI_Z = 0x00000002,

     DIAI_FIRST = DIAI_X,

     DIAI_LAST = 0x0000001F
} DFBInputDeviceAxisIdentifier;




typedef enum {
     DWDESC_CAPS = 0x00000001,
     DWDESC_WIDTH = 0x00000002,
     DWDESC_HEIGHT = 0x00000004,
     DWDESC_PIXELFORMAT = 0x00000008,
     DWDESC_POSX = 0x00000010,
     DWDESC_POSY = 0x00000020,
     DWDESC_SURFACE_CAPS = 0x00000040

} DFBWindowDescriptionFlags;




typedef enum {
     DBDESC_FILE = 0x00000001,

     DBDESC_MEMORY = 0x00000002

} DFBDataBufferDescriptionFlags;




typedef enum {
     DWCAPS_NONE = 0x00000000,
     DWCAPS_ALPHACHANNEL = 0x00000001,

     DWCAPS_DOUBLEBUFFER = 0x00000002,







     DWCAPS_INPUTONLY = 0x00000004,


     DWCAPS_NODECORATION = 0x00000008,
     DWCAPS_ALL = 0x0000000F
} DFBWindowCapabilities;
# 840 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DFFA_NONE = 0x00000000,
     DFFA_NOKERNING = 0x00000001,
     DFFA_NOHINTING = 0x00000002,
     DFFA_MONOCHROME = 0x00000004,
     DFFA_NOCHARMAP = 0x00000008

} DFBFontAttributes;




typedef enum {
     DFDESC_ATTRIBUTES = 0x00000001,
     DFDESC_HEIGHT = 0x00000002,
     DFDESC_WIDTH = 0x00000004,
     DFDESC_INDEX = 0x00000008,
     DFDESC_FIXEDADVANCE = 0x00000010,


     DFDESC_FRACT_HEIGHT = 0x00000020,
     DFDESC_FRACT_WIDTH = 0x00000040,
} DFBFontDescriptionFlags;
# 880 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef struct {
     DFBFontDescriptionFlags flags;

     DFBFontAttributes attributes;
     int height;
     int width;
     unsigned int index;
     int fixed_advance;

     int fract_height;
     int fract_width;
} DFBFontDescription;
# 930 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DSPF_UNKNOWN = 0x00000000,


     DSPF_ARGB1555 = ( (((0 ) & 0x7F) ) | (((15) & 0x1F) << 7) | (((1) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB16 = ( (((1 ) & 0x7F) ) | (((16) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB24 = ( (((2 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((3 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB32 = ( (((3 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((4 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ARGB = ( (((4 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((8) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((4 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_A8 = ( (((5 ) & 0x7F) ) | (((0) & 0x1F) << 7) | (((8) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_YUY2 = ( (((6 ) & 0x7F) ) | (((16) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB332 = ( (((7 ) & 0x7F) ) | (((8) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_UYVY = ( (((8 ) & 0x7F) ) | (((16) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_I420 = ( (((9 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((2 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_YV12 = ( (((10 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((2 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_LUT8 = ( (((11 ) & 0x7F) ) | (((8) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((1 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ALUT44 = ( (((12 ) & 0x7F) ) | (((4) & 0x1F) << 7) | (((4) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((1 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_AiRGB = ( (((13 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((8) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((4 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((1 ) ? 1 :0) << 31) ),


     DSPF_A1 = ( (((14 ) & 0x7F) ) | (((0) & 0x1F) << 7) | (((1) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((1 ) & 0x07) << 17) | (((0 ) & 0x07) << 20) | (((7 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_NV12 = ( (((15 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((2 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_NV16 = ( (((16 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((1 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ARGB2554 = ( (((17 ) & 0x7F) ) | (((14) & 0x1F) << 7) | (((2) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ARGB4444 = ( (((18 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((4) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_NV21 = ( (((19 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((2 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_AYUV = ( (((20 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((8) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((4 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_A4 = ( (((21 ) & 0x7F) ) | (((0) & 0x1F) << 7) | (((4) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((4 ) & 0x07) << 17) | (((0 ) & 0x07) << 20) | (((1 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ARGB1666 = ( (((22 ) & 0x7F) ) | (((18) & 0x1F) << 7) | (((1) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((3 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ARGB6666 = ( (((23 ) & 0x7F) ) | (((18) & 0x1F) << 7) | (((6) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((3 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB18 = ( (((24 ) & 0x7F) ) | (((18) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((3 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_LUT2 = ( (((25 ) & 0x7F) ) | (((2) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((2 ) & 0x07) << 17) | (((0 ) & 0x07) << 20) | (((3 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((1 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_XRGB4444 = ( (((26 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_XRGB1555 = ( (((27 ) & 0x7F) ) | (((15) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) )

} DFBSurfacePixelFormat;
# 1051 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef struct {
     DFBSurfaceDescriptionFlags flags;

     DFBSurfaceCapabilities caps;
     int width;
     int height;
     DFBSurfacePixelFormat pixelformat;

     struct {
          void *data;
          int pitch;
     } preallocated[2];

     struct {
          const DFBColor *entries;
          unsigned int size;
     } palette;
} DFBSurfaceDescription;




typedef struct {
     DFBPaletteDescriptionFlags flags;

     DFBPaletteCapabilities caps;
     unsigned int size;
     const DFBColor *entries;

} DFBPaletteDescription;







typedef struct {
     DFBDisplayLayerTypeFlags type;
     DFBDisplayLayerCapabilities caps;

     char name[32];

     int level;
     int regions;



     int sources;
     int clip_regions;
} DFBDisplayLayerDescription;







typedef struct {
     DFBDisplayLayerSourceID source_id;

     char name[24];
} DFBDisplayLayerSourceDescription;







typedef struct {
     DFBScreenCapabilities caps;


     char name[32];

     int mixers;

     int encoders;

     int outputs;

} DFBScreenDescription;
# 1142 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef struct {
     DFBInputDeviceTypeFlags type;

     DFBInputDeviceCapabilities caps;



     int min_keycode;




     int max_keycode;




     DFBInputDeviceAxisIdentifier max_axis;

     DFBInputDeviceButtonIdentifier max_button;


     char name[32];

     char vendor[40];
} DFBInputDeviceDescription;





typedef struct {
     int major;
     int minor;

     char name[40];
     char vendor[60];
} DFBGraphicsDriverInfo;







typedef struct {
     DFBAccelerationMask acceleration_mask;

     DFBSurfaceBlittingFlags blitting_flags;
     DFBSurfaceDrawingFlags drawing_flags;

     unsigned int video_memory;

     char name[48];
     char vendor[64];

     DFBGraphicsDriverInfo driver;
} DFBGraphicsDeviceDescription;




typedef struct {
     DFBWindowDescriptionFlags flags;

     DFBWindowCapabilities caps;
     int width;
     int height;
     DFBSurfacePixelFormat pixelformat;
     int posx;
     int posy;
     DFBSurfaceCapabilities surface_caps;
} DFBWindowDescription;




typedef struct {
     DFBDataBufferDescriptionFlags flags;

     const char *file;

     struct {
          const void *data;
          unsigned int length;
     } memory;
} DFBDataBufferDescription;




typedef enum {
     DFENUM_OK = 0x00000000,
     DFENUM_CANCEL = 0x00000001
} DFBEnumerationResult;




typedef DFBEnumerationResult (*DFBVideoModeCallback) (
     int width,
     int height,
     int bpp,
     void *callbackdata
);





typedef DFBEnumerationResult (*DFBScreenCallback) (
     DFBScreenID screen_id,
     DFBScreenDescription desc,
     void *callbackdata
);





typedef DFBEnumerationResult (*DFBDisplayLayerCallback) (
     DFBDisplayLayerID layer_id,
     DFBDisplayLayerDescription desc,
     void *callbackdata
);





typedef DFBEnumerationResult (*DFBInputDeviceCallback) (
     DFBInputDeviceID device_id,
     DFBInputDeviceDescription desc,
     void *callbackdata
);







typedef int (*DFBGetDataCallback) (
     void *buffer,
     unsigned int length,
     void *callbackdata
);




typedef enum {
     DVCAPS_BASIC = 0x00000000,
     DVCAPS_SEEK = 0x00000001,
     DVCAPS_SCALE = 0x00000002,
     DVCAPS_INTERLACED = 0x00000004,
     DVCAPS_SPEED = 0x00000008,
     DVCAPS_BRIGHTNESS = 0x00000010,
     DVCAPS_CONTRAST = 0x00000020,
     DVCAPS_HUE = 0x00000040,
     DVCAPS_SATURATION = 0x00000080,
     DVCAPS_INTERACTIVE = 0x00000100,
     DVCAPS_VOLUME = 0x00000200,
     DVCAPS_EVENT = 0x00000400,
     DVCAPS_ATTRIBUTES = 0x00000800,
     DVCAPS_AUDIO_SEL = 0x00001000,
} DFBVideoProviderCapabilities;




typedef enum {
     DVSTATE_UNKNOWN = 0x00000000,
     DVSTATE_PLAY = 0x00000001,
     DVSTATE_STOP = 0x00000002,
     DVSTATE_FINISHED = 0x00000003,
     DVSTATE_BUFFERING = 0x00000004

} DFBVideoProviderStatus;




typedef enum {
     DVPLAY_NOFX = 0x00000000,
     DVPLAY_REWIND = 0x00000001,
     DVPLAY_LOOPING = 0x00000002


} DFBVideoProviderPlaybackFlags;




typedef enum {
     DVAUDIOUNIT_NONE = 0x00000000,
     DVAUDIOUNIT_ONE = 0x00000001,
     DVAUDIOUNIT_TWO = 0x00000002,
     DVAUDIOUNIT_THREE = 0x00000004,
     DVAUDIOUNIT_FOUR = 0x00000008,
     DVAUDIOUNIT_ALL = 0x0000000F,
} DFBVideoProviderAudioUnits;





typedef enum {
     DCAF_NONE = 0x00000000,
     DCAF_BRIGHTNESS = 0x00000001,
     DCAF_CONTRAST = 0x00000002,
     DCAF_HUE = 0x00000004,
     DCAF_SATURATION = 0x00000008,
     DCAF_GAMMA = 0x00000010,
     DCAF_ALL = 0x0000001F
} DFBColorAdjustmentFlags;







typedef struct {
     DFBColorAdjustmentFlags flags;

     u16 brightness;
     u16 contrast;
     u16 hue;
     u16 saturation;
     double gamma;
} DFBColorAdjustment;
# 1434 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFB { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFB *thiz ); DFBResult (*Release)( IDirectFB *thiz ); DFBResult (*SetCooperativeLevel) ( IDirectFB *thiz, DFBCooperativeLevel level ); DFBResult (*SetVideoMode) ( IDirectFB *thiz, int width, int height, int bpp ); DFBResult (*GetDeviceDescription) ( IDirectFB *thiz, DFBGraphicsDeviceDescription *ret_desc ); DFBResult (*EnumVideoModes) ( IDirectFB *thiz, DFBVideoModeCallback callback, void *callbackdata ); DFBResult (*CreateSurface) ( IDirectFB *thiz, const DFBSurfaceDescription *desc, IDirectFBSurface **ret_interface ); DFBResult (*CreatePalette) ( IDirectFB *thiz, const DFBPaletteDescription *desc, IDirectFBPalette **ret_interface ); DFBResult (*EnumScreens) ( IDirectFB *thiz, DFBScreenCallback callback, void *callbackdata ); DFBResult (*GetScreen) ( IDirectFB *thiz, DFBScreenID screen_id, IDirectFBScreen **ret_interface ); DFBResult (*EnumDisplayLayers) ( IDirectFB *thiz, DFBDisplayLayerCallback callback, void *callbackdata ); DFBResult (*GetDisplayLayer) ( IDirectFB *thiz, DFBDisplayLayerID layer_id, IDirectFBDisplayLayer **ret_interface ); DFBResult (*EnumInputDevices) ( IDirectFB *thiz, DFBInputDeviceCallback callback, void *callbackdata ); DFBResult (*GetInputDevice) ( IDirectFB *thiz, DFBInputDeviceID device_id, IDirectFBInputDevice **ret_interface ); DFBResult (*CreateEventBuffer) ( IDirectFB *thiz, IDirectFBEventBuffer **ret_buffer ); DFBResult (*CreateInputEventBuffer) ( IDirectFB *thiz, DFBInputDeviceCapabilities caps, DFBBoolean global, IDirectFBEventBuffer **ret_buffer ); DFBResult (*CreateImageProvider) ( IDirectFB *thiz, const char *filename, IDirectFBImageProvider **ret_interface ); DFBResult (*CreateVideoProvider) ( IDirectFB *thiz, const char *filename, IDirectFBVideoProvider **ret_interface ); DFBResult (*CreateFont) ( IDirectFB *thiz, const char *filename, const DFBFontDescription *desc, IDirectFBFont **ret_interface ); DFBResult (*CreateDataBuffer) ( IDirectFB *thiz, const DFBDataBufferDescription *desc, IDirectFBDataBuffer **ret_interface ); DFBResult (*SetClipboardData) ( IDirectFB *thiz, const char *mime_type, const void *data, unsigned int size, struct timeval *ret_timestamp ); DFBResult (*GetClipboardData) ( IDirectFB *thiz, char **ret_mimetype, void **ret_data, unsigned int *ret_size ); DFBResult (*GetClipboardTimeStamp) ( IDirectFB *thiz, struct timeval *ret_timestamp ); DFBResult (*Suspend) ( IDirectFB *thiz ); DFBResult (*Resume) ( IDirectFB *thiz ); DFBResult (*WaitIdle) ( IDirectFB *thiz ); DFBResult (*WaitForSync) ( IDirectFB *thiz ); DFBResult (*GetInterface) ( IDirectFB *thiz, const char *type, const char *implementation, void *arg, void **ret_interface ); };
# 1797 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DLSCL_SHARED = 0,
     DLSCL_EXCLUSIVE,

     DLSCL_ADMINISTRATIVE

} DFBDisplayLayerCooperativeLevel;





typedef enum {
     DLBM_DONTCARE = 0,

     DLBM_COLOR,

     DLBM_IMAGE,
     DLBM_TILE
} DFBDisplayLayerBackgroundMode;




typedef enum {
     DLCONF_NONE = 0x00000000,

     DLCONF_WIDTH = 0x00000001,
     DLCONF_HEIGHT = 0x00000002,
     DLCONF_PIXELFORMAT = 0x00000004,
     DLCONF_BUFFERMODE = 0x00000008,
     DLCONF_OPTIONS = 0x00000010,
     DLCONF_SOURCE = 0x00000020,
     DLCONF_SURFACE_CAPS = 0x00000040,

     DLCONF_ALL = 0x0000007F
} DFBDisplayLayerConfigFlags;




typedef struct {
     DFBDisplayLayerConfigFlags flags;

     int width;
     int height;
     DFBSurfacePixelFormat pixelformat;
     DFBDisplayLayerBufferMode buffermode;
     DFBDisplayLayerOptions options;
     DFBDisplayLayerSourceID source;

     DFBSurfaceCapabilities surface_caps;

} DFBDisplayLayerConfig;




typedef enum {
     DSPM_ON = 0,
     DSPM_STANDBY,
     DSPM_SUSPEND,
     DSPM_OFF
} DFBScreenPowerMode;





typedef enum {
     DSMCAPS_NONE = 0x00000000,

     DSMCAPS_FULL = 0x00000001,
     DSMCAPS_SUB_LEVEL = 0x00000002,
     DSMCAPS_SUB_LAYERS = 0x00000004,
     DSMCAPS_BACKGROUND = 0x00000008
} DFBScreenMixerCapabilities;







typedef struct {
     DFBScreenMixerCapabilities caps;

     DFBDisplayLayerIDs layers;


     int sub_num;

     DFBDisplayLayerIDs sub_layers;


     char name[24];
} DFBScreenMixerDescription;




typedef enum {
     DSMCONF_NONE = 0x00000000,

     DSMCONF_TREE = 0x00000001,
     DSMCONF_LEVEL = 0x00000002,
     DSMCONF_LAYERS = 0x00000004,

     DSMCONF_BACKGROUND = 0x00000010,

     DSMCONF_ALL = 0x00000017
} DFBScreenMixerConfigFlags;




typedef enum {
     DSMT_UNKNOWN = 0x00000000,

     DSMT_FULL = 0x00000001,
     DSMT_SUB_LEVEL = 0x00000002,
     DSMT_SUB_LAYERS = 0x00000003
} DFBScreenMixerTree;




typedef struct {
     DFBScreenMixerConfigFlags flags;

     DFBScreenMixerTree tree;

     int level;
     DFBDisplayLayerIDs layers;

     DFBColor background;
} DFBScreenMixerConfig;





typedef enum {
     DSOCAPS_NONE = 0x00000000,

     DSOCAPS_CONNECTORS = 0x00000001,

     DSOCAPS_ENCODER_SEL = 0x00000010,
     DSOCAPS_SIGNAL_SEL = 0x00000020,
     DSOCAPS_CONNECTOR_SEL = 0x00000040,
     DSOCAPS_SLOW_BLANKING = 0x00000080,
     DSOCAPS_RESOLUTION = 0x00000100,
     DSOCAPS_ALL = 0x000001F1
} DFBScreenOutputCapabilities;




typedef enum {
     DSOC_UNKNOWN = 0x00000000,

     DSOC_VGA = 0x00000001,
     DSOC_SCART = 0x00000002,
     DSOC_YC = 0x00000004,
     DSOC_CVBS = 0x00000008,
     DSOC_SCART2 = 0x00000010,
     DSOC_COMPONENT = 0x00000020,
     DSOC_HDMI = 0x00000040
} DFBScreenOutputConnectors;




typedef enum {
     DSOS_NONE = 0x00000000,

     DSOS_VGA = 0x00000001,
     DSOS_YC = 0x00000002,
     DSOS_CVBS = 0x00000004,
     DSOS_RGB = 0x00000008,
     DSOS_YCBCR = 0x00000010,
     DSOS_HDMI = 0x00000020,
     DSOS_656 = 0x00000040
} DFBScreenOutputSignals;





typedef enum {
     DSOSB_OFF = 0x00000000,
     DSOSB_16x9 = 0x00000001,
     DSOSB_4x3 = 0x00000002,
     DSOSB_FOLLOW = 0x00000004,
     DSOSB_MONITOR = 0x00000008
} DFBScreenOutputSlowBlankingSignals;





typedef enum {
    DSOR_UNKNOWN = 0x00000000,
    DSOR_640_480 = 0x00000001,
    DSOR_720_480 = 0x00000002,
    DSOR_720_576 = 0x00000004,
    DSOR_800_600 = 0x00000008,
    DSOR_1024_768 = 0x00000010,
    DSOR_1152_864 = 0x00000020,
    DSOR_1280_720 = 0x00000040,
    DSOR_1280_768 = 0x00000080,
    DSOR_1280_960 = 0x00000100,
    DSOR_1280_1024 = 0x00000200,
    DSOR_1400_1050 = 0x00000400,
    DSOR_1600_1200 = 0x00000800,
    DSOR_1920_1080 = 0x00001000,
    DSOR_ALL = 0x00001FFF
} DFBScreenOutputResolution;






typedef struct {
     DFBScreenOutputCapabilities caps;

     DFBScreenOutputConnectors all_connectors;
     DFBScreenOutputSignals all_signals;
     DFBScreenOutputResolution all_resolutions;

     char name[24];
} DFBScreenOutputDescription;




typedef enum {
     DSOCONF_NONE = 0x00000000,

     DSOCONF_ENCODER = 0x00000001,
     DSOCONF_SIGNALS = 0x00000002,
     DSOCONF_CONNECTORS = 0x00000004,
     DSOCONF_SLOW_BLANKING= 0x00000008,
     DSOCONF_RESOLUTION = 0x00000010,

     DSOCONF_ALL = 0x0000001F
} DFBScreenOutputConfigFlags;




typedef struct {
     DFBScreenOutputConfigFlags flags;

     int encoder;
     DFBScreenOutputSignals out_signals;
     DFBScreenOutputConnectors out_connectors;
     DFBScreenOutputSlowBlankingSignals slow_blanking;
     DFBScreenOutputResolution resolution;
} DFBScreenOutputConfig;





typedef enum {
     DSECAPS_NONE = 0x00000000,

     DSECAPS_TV_STANDARDS = 0x00000001,
     DSECAPS_TEST_PICTURE = 0x00000002,
     DSECAPS_MIXER_SEL = 0x00000004,
     DSECAPS_OUT_SIGNALS = 0x00000008,
     DSECAPS_SCANMODE = 0x00000010,
     DSECAPS_FREQUENCY = 0x00000020,

     DSECAPS_BRIGHTNESS = 0x00000100,
     DSECAPS_CONTRAST = 0x00000200,
     DSECAPS_HUE = 0x00000400,
     DSECAPS_SATURATION = 0x00000800,

     DSECAPS_CONNECTORS = 0x00001000,
     DSECAPS_SLOW_BLANKING = 0x00002000,

     DSECAPS_RESOLUTION = 0x00004000,
     DSECAPS_WSS = 0x00008000,
     DSECAPS_MACROVISION = 0x00010000,
     DSECAPS_CGMS = 0x00020000,

     DSECAPS_ALL = 0x0003ff3f
} DFBScreenEncoderCapabilities;




typedef enum
{
    DSEWAR_ASPECT_RATIO_FF_4TO3 = 0x00000000,
    DSEWAR_ASPECT_RATIO_LB_14TO9_CTR = 0x00000001,
    DSEWAR_ASPECT_RATIO_LB_14TO9_TOP = 0x00000002,
    DSEWAR_ASPECT_RATIO_LB_16TO9_CTR = 0x00000004,
    DSEWAR_ASPECT_RATIO_LB_16TO9_TOP = 0x00000008,
    DSEWAR_ASPECT_RATIO_LB_P16TO9_CTR = 0x00000010,
    DSEWAR_ASPECT_RATIO_FF_14TO9_CTR = 0x00000020,
    DSEWAR_ASPECT_RATIO_FF_16TO9 = 0x00000040,
} DFBScreenEncoderWssAspectRatio;




typedef enum
{
    DSEWM_MODE_CAMERA = 0x00000000,
    DSEWM_MODE_FILM = 0x00000001
} DFBScreenEncoderWssMode;




typedef enum
{
    DSEWCC_STANDARD_CODING = 0x00000000,
    DSEWCC_MOTION_ADAPTATIVE_COLOUR_PLUS = 0x00000001
} DFBScreenEncoderWssColourCoding;




typedef enum
{
    DSEWH_NO_HELPER = 0x00000000,
    DSEWH_MODULATED_HELPER = 0x00000001
} DFBScreenEncoderWssHelper;




typedef enum
{
    DSEWS_NO_SUBTITLES_WITHIN_TELETEXT = 0x00000000,
    DSEWS_SUBTITLES_WITHIN_TELETEXT = 0x00000001
} DFBScreenEncoderWssSubtitles;




typedef enum
{
    DSEWSM_SBT_TYPE_NO_OPEN = 0x00000000,
    DSEWSM_SBT_TYPE_INSIDE_IMAGE = 0x00000001,
    DSEWSM_SBT_TYPE_OUTSIDE_IMAGE = 0x00000002,
    DSEWSM_SBT_TYPE_RESERVED = 0x00000004
} DFBScreenEncoderWssSubtitlingMode;




typedef enum
{
    DSEWSS_NO_SURROUND_SOUND = 0x00000000,
    DSEWSS_SURROUND_SOUND_MODE = 0x00000001
} DFBScreenEncoderWssSurroundSound;




typedef enum
{
    DSEWC_NO_COPYRIGHT_OR_UNKNOWN = 0x00000000,
    DSEWC_COPYRIGHT_ASSERTED = 0x00000001
} DFBScreenEncoderWssCopyright;




typedef enum
{
    DSEWG_COPYING_NOT_RESTRICTED = 0x00000000,
    DSEWG_COPYING_RESTRICTED = 0x00000001
} DFBScreenEncoderWssGeneration;




typedef enum
{
    DSEWG_DISABLE = 0x00000000,
    DSEWG_ENABLE = 0x00000001
} DFBScreenEncoderWssEnable;




typedef struct {
    DFBScreenEncoderWssEnable enable;
 DFBScreenEncoderWssAspectRatio aspect_ratio;
 DFBScreenEncoderWssMode mode;
 DFBScreenEncoderWssColourCoding colour;
 DFBScreenEncoderWssHelper helper;
 DFBScreenEncoderWssSubtitles sbt;
 DFBScreenEncoderWssSubtitlingMode sbt_type;
 DFBScreenEncoderWssSurroundSound surround;
 DFBScreenEncoderWssCopyright copyright;
 DFBScreenEncoderWssGeneration generation;
} DFBScreenEncoderWideScreenSignaling;




typedef enum {
     DSEM_OFF = 0x00000000,
     DSEM_AGC_AND_2L_CSP = 0x00000001,
     DSEM_AGC_AND_4L_CSP = 0x00000002,
     DSEM_AGC_ONLY = 0x00000004
} DFBScreenEncoderMacrovision;




typedef enum
{
    DSECGMS_OFF = 0x00000000,
    DSECGMS_NO_COPY_RESTRICTION = 0x00000001,
    DSECGMS_COPY_NO_MORE = 0x00000002,
    DSECGMS_COPY_ONCE_ALLOWED = 0x00000004,
    DSECGMS_NO_COPYING_PERMITTED = 0x00000008
} DFBScreenEncoderCopyGenerationManagementSystem;




typedef enum {
     DSET_UNKNOWN = 0x00000000,

     DSET_CRTC = 0x00000001,
     DSET_TV = 0x00000002,
     DSET_DIGITAL = 0x00000004
} DFBScreenEncoderType;




typedef enum {
     DSETV_UNKNOWN = 0x00000000,

     DSETV_PAL = 0x00000001,
     DSETV_NTSC = 0x00000002,
     DSETV_SECAM = 0x00000004,
     DSETV_PAL_60 = 0x00000008,
     DSETV_PAL_BG = 0x00000010,
     DSETV_PAL_I = 0x00000020,
     DSETV_PAL_M = 0x00000040,
     DSETV_PAL_N = 0x00000080,
     DSETV_PAL_NC = 0x00000100,
     DSETV_NTSC_M_JPN = 0x00000200,
     DSETV_DIGITAL = 0x00000400,
     DSETV_ALL = 0x000007FF
} DFBScreenEncoderTVStandards;




typedef enum {
     DSESM_UNKNOWN = 0x00000000,

     DSESM_INTERLACED = 0x00000001,
     DSESM_PROGRESSIVE = 0x00000002
} DFBScreenEncoderScanMode;





typedef enum {
     DSEF_UNKNOWN = 0x00000000,

     DSEF_25HZ = 0x00000001,
     DSEF_29_97HZ = 0x00000002,
     DSEF_50HZ = 0x00000004,
     DSEF_59_94HZ = 0x00000008,
     DSEF_60HZ = 0x00000010,
     DSEF_75HZ = 0x00000020,
} DFBScreenEncoderFrequency;






typedef struct {
     DFBScreenEncoderCapabilities caps;
     DFBScreenEncoderType type;

     DFBScreenEncoderTVStandards tv_standards;
     DFBScreenOutputSignals out_signals;
     DFBScreenOutputConnectors all_connectors;

     DFBScreenOutputResolution all_resolutions;
     char name[24];
} DFBScreenEncoderDescription;




typedef enum {
     DSECONF_NONE = 0x00000000,

     DSECONF_TV_STANDARD = 0x00000001,
     DSECONF_TEST_PICTURE = 0x00000002,
     DSECONF_MIXER = 0x00000004,
     DSECONF_OUT_SIGNALS = 0x00000008,
     DSECONF_SCANMODE = 0x00000010,
     DSECONF_TEST_COLOR = 0x00000020,
     DSECONF_ADJUSTMENT = 0x00000040,

     DSECONF_FREQUENCY = 0x00000080,
     DSECONF_CONNECTORS = 0x00000100,
     DSECONF_SLOW_BLANKING = 0x00000200,
     DSECONF_RESOLUTION = 0x00000400,
     DSECONF_WSS = 0x00000800,
     DSECONF_MACROVISION = 0x00001000,
     DSECONF_CGMS = 0x00002000,

     DSECONF_ALL = 0x00003FFF
} DFBScreenEncoderConfigFlags;




typedef enum {
     DSETP_OFF = 0x00000000,

     DSETP_MULTI = 0x00000001,
     DSETP_SINGLE = 0x00000002,

     DSETP_WHITE = 0x00000010,
     DSETP_YELLOW = 0x00000020,
     DSETP_CYAN = 0x00000030,
     DSETP_GREEN = 0x00000040,
     DSETP_MAGENTA = 0x00000050,
     DSETP_RED = 0x00000060,
     DSETP_BLUE = 0x00000070,
     DSETP_BLACK = 0x00000080
} DFBScreenEncoderTestPicture;




typedef struct {
     DFBScreenEncoderConfigFlags flags;

     DFBScreenEncoderTVStandards tv_standard;
     DFBScreenEncoderTestPicture test_picture;
     int mixer;
     DFBScreenOutputSignals out_signals;
     DFBScreenOutputConnectors out_connectors;
     DFBScreenOutputSlowBlankingSignals slow_blanking;

     DFBScreenEncoderWideScreenSignaling wss;
     DFBScreenEncoderMacrovision macrovision;
     DFBScreenEncoderCopyGenerationManagementSystem cgms;
     DFBScreenEncoderScanMode scanmode;

     DFBColor test_color;

     DFBColorAdjustment adjustment;
     DFBScreenEncoderFrequency frequency;

     DFBScreenOutputResolution resolution;
} DFBScreenEncoderConfig;
# 2376 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFBScreen { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFBScreen *thiz ); DFBResult (*Release)( IDirectFBScreen *thiz ); DFBResult (*GetID) ( IDirectFBScreen *thiz, DFBScreenID *ret_screen_id ); DFBResult (*GetDescription) ( IDirectFBScreen *thiz, DFBScreenDescription *ret_desc ); DFBResult (*GetSize) ( IDirectFBScreen *thiz, int *ret_width, int *ret_height ); DFBResult (*EnumDisplayLayers) ( IDirectFBScreen *thiz, DFBDisplayLayerCallback callback, void *callbackdata ); DFBResult (*SetPowerMode) ( IDirectFBScreen *thiz, DFBScreenPowerMode mode ); DFBResult (*WaitForSync) ( IDirectFBScreen *thiz ); DFBResult (*GetMixerDescriptions) ( IDirectFBScreen *thiz, DFBScreenMixerDescription *ret_descriptions ); DFBResult (*GetMixerConfiguration) ( IDirectFBScreen *thiz, int mixer, DFBScreenMixerConfig *ret_config ); DFBResult (*TestMixerConfiguration) ( IDirectFBScreen *thiz, int mixer, const DFBScreenMixerConfig *config, DFBScreenMixerConfigFlags *ret_failed ); DFBResult (*SetMixerConfiguration) ( IDirectFBScreen *thiz, int mixer, const DFBScreenMixerConfig *config ); DFBResult (*GetEncoderDescriptions) ( IDirectFBScreen *thiz, DFBScreenEncoderDescription *ret_descriptions ); DFBResult (*GetEncoderConfiguration) ( IDirectFBScreen *thiz, int encoder, DFBScreenEncoderConfig *ret_config ); DFBResult (*TestEncoderConfiguration) ( IDirectFBScreen *thiz, int encoder, const DFBScreenEncoderConfig *config, DFBScreenEncoderConfigFlags *ret_failed ); DFBResult (*SetEncoderConfiguration) ( IDirectFBScreen *thiz, int encoder, const DFBScreenEncoderConfig *config ); DFBResult (*GetOutputDescriptions) ( IDirectFBScreen *thiz, DFBScreenOutputDescription *ret_descriptions ); DFBResult (*GetOutputConfiguration) ( IDirectFBScreen *thiz, int output, DFBScreenOutputConfig *ret_config ); DFBResult (*TestOutputConfiguration) ( IDirectFBScreen *thiz, int output, const DFBScreenOutputConfig *config, DFBScreenOutputConfigFlags *ret_failed ); DFBResult (*SetOutputConfiguration) ( IDirectFBScreen *thiz, int output, const DFBScreenOutputConfig *config ); };
# 2591 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFBDisplayLayer { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFBDisplayLayer *thiz ); DFBResult (*Release)( IDirectFBDisplayLayer *thiz ); DFBResult (*GetID) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerID *ret_layer_id ); DFBResult (*GetDescription) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerDescription *ret_desc ); DFBResult (*GetSourceDescriptions) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerSourceDescription *ret_descriptions ); DFBResult (*GetCurrentOutputField) ( IDirectFBDisplayLayer *thiz, int *ret_field ); DFBResult (*GetSurface) ( IDirectFBDisplayLayer *thiz, IDirectFBSurface **ret_interface ); DFBResult (*GetScreen) ( IDirectFBDisplayLayer *thiz, IDirectFBScreen **ret_interface ); DFBResult (*SetCooperativeLevel) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerCooperativeLevel level ); DFBResult (*GetConfiguration) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerConfig *ret_config ); DFBResult (*TestConfiguration) ( IDirectFBDisplayLayer *thiz, const DFBDisplayLayerConfig *config, DFBDisplayLayerConfigFlags *ret_failed ); DFBResult (*SetConfiguration) ( IDirectFBDisplayLayer *thiz, const DFBDisplayLayerConfig *config ); DFBResult (*SetScreenLocation) ( IDirectFBDisplayLayer *thiz, float x, float y, float width, float height ); DFBResult (*SetScreenPosition) ( IDirectFBDisplayLayer *thiz, int x, int y ); DFBResult (*SetScreenRectangle) ( IDirectFBDisplayLayer *thiz, int x, int y, int width, int height ); DFBResult (*SetOpacity) ( IDirectFBDisplayLayer *thiz, u8 opacity ); DFBResult (*SetSourceRectangle) ( IDirectFBDisplayLayer *thiz, int x, int y, int width, int height ); DFBResult (*SetFieldParity) ( IDirectFBDisplayLayer *thiz, int field ); DFBResult (*SetClipRegions) ( IDirectFBDisplayLayer *thiz, const DFBRegion *regions, int num_regions, DFBBoolean positive ); DFBResult (*SetSrcColorKey) ( IDirectFBDisplayLayer *thiz, u8 r, u8 g, u8 b ); DFBResult (*SetDstColorKey) ( IDirectFBDisplayLayer *thiz, u8 r, u8 g, u8 b ); DFBResult (*GetLevel) ( IDirectFBDisplayLayer *thiz, int *ret_level ); DFBResult (*SetLevel) ( IDirectFBDisplayLayer *thiz, int level ); DFBResult (*SetBackgroundMode) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerBackgroundMode mode ); DFBResult (*SetBackgroundImage) ( IDirectFBDisplayLayer *thiz, IDirectFBSurface *surface ); DFBResult (*SetBackgroundColor) ( IDirectFBDisplayLayer *thiz, u8 r, u8 g, u8 b, u8 a ); DFBResult (*GetColorAdjustment) ( IDirectFBDisplayLayer *thiz, DFBColorAdjustment *ret_adj ); DFBResult (*SetColorAdjustment) ( IDirectFBDisplayLayer *thiz, const DFBColorAdjustment *adj ); DFBResult (*CreateWindow) ( IDirectFBDisplayLayer *thiz, const DFBWindowDescription *desc, IDirectFBWindow **ret_interface ); DFBResult (*GetWindow) ( IDirectFBDisplayLayer *thiz, DFBWindowID window_id, IDirectFBWindow **ret_interface ); DFBResult (*EnableCursor) ( IDirectFBDisplayLayer *thiz, int enable ); DFBResult (*GetCursorPosition) ( IDirectFBDisplayLayer *thiz, int *ret_x, int *ret_y ); DFBResult (*WarpCursor) ( IDirectFBDisplayLayer *thiz, int x, int y ); DFBResult (*SetCursorAcceleration) ( IDirectFBDisplayLayer *thiz, int numerator, int denominator, int threshold ); DFBResult (*SetCursorShape) ( IDirectFBDisplayLayer *thiz, IDirectFBSurface *shape, int hot_x, int hot_y ); DFBResult (*SetCursorOpacity) ( IDirectFBDisplayLayer *thiz, u8 opacity ); DFBResult (*WaitForSync) ( IDirectFBDisplayLayer *thiz ); DFBResult (*SwitchContext) ( IDirectFBDisplayLayer *thiz, DFBBoolean exclusive ); };
# 3038 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DSFLIP_NONE = 0x00000000,

     DSFLIP_WAIT = 0x00000001,

     DSFLIP_BLIT = 0x00000002,



     DSFLIP_ONSYNC = 0x00000004,



     DSFLIP_PIPELINE = 0x00000008,

     DSFLIP_WAITFORSYNC = DSFLIP_WAIT | DSFLIP_ONSYNC
} DFBSurfaceFlipFlags;




typedef enum {
     DSTF_LEFT = 0x00000000,
     DSTF_CENTER = 0x00000001,
     DSTF_RIGHT = 0x00000002,

     DSTF_TOP = 0x00000004,

     DSTF_BOTTOM = 0x00000008,


     DSTF_TOPLEFT = DSTF_TOP | DSTF_LEFT,
     DSTF_TOPCENTER = DSTF_TOP | DSTF_CENTER,
     DSTF_TOPRIGHT = DSTF_TOP | DSTF_RIGHT,

     DSTF_BOTTOMLEFT = DSTF_BOTTOM | DSTF_LEFT,
     DSTF_BOTTOMCENTER = DSTF_BOTTOM | DSTF_CENTER,
     DSTF_BOTTOMRIGHT = DSTF_BOTTOM | DSTF_RIGHT
} DFBSurfaceTextFlags;





typedef enum {
     DSLF_READ = 0x00000001,

     DSLF_WRITE = 0x00000002
} DFBSurfaceLockFlags;




typedef enum {



     DSPD_NONE = 0,
     DSPD_CLEAR = 1,
     DSPD_SRC = 2,
     DSPD_SRC_OVER = 3,
     DSPD_DST_OVER = 4,
     DSPD_SRC_IN = 5,
     DSPD_DST_IN = 6,
     DSPD_SRC_OUT = 7,
     DSPD_DST_OUT = 8,
     DSPD_SRC_ATOP = 9,
     DSPD_DST_ATOP = 10,
     DSPD_ADD = 11,
     DSPD_XOR = 12,
} DFBSurfacePorterDuffRule;




typedef enum {
     DSBF_ZERO = 1,
     DSBF_ONE = 2,
     DSBF_SRCCOLOR = 3,
     DSBF_INVSRCCOLOR = 4,
     DSBF_SRCALPHA = 5,
     DSBF_INVSRCALPHA = 6,
     DSBF_DESTALPHA = 7,
     DSBF_INVDESTALPHA = 8,
     DSBF_DESTCOLOR = 9,
     DSBF_INVDESTCOLOR = 10,
     DSBF_SRCALPHASAT = 11
} DFBSurfaceBlendFunction;




typedef struct {
     float x;
     float y;
     float z;
     float w;

     float s;
     float t;
} DFBVertex;




typedef enum {
     DTTF_LIST,
     DTTF_STRIP,
     DTTF_FAN
} DFBTriangleFormation;
# 3156 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFBSurface { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFBSurface *thiz ); DFBResult (*Release)( IDirectFBSurface *thiz ); DFBResult (*GetCapabilities) ( IDirectFBSurface *thiz, DFBSurfaceCapabilities *ret_caps ); DFBResult (*GetPosition) ( IDirectFBSurface *thiz, int *ret_x, int *ret_y ); DFBResult (*GetSize) ( IDirectFBSurface *thiz, int *ret_width, int *ret_height ); DFBResult (*GetVisibleRectangle) ( IDirectFBSurface *thiz, DFBRectangle *ret_rect ); DFBResult (*GetPixelFormat) ( IDirectFBSurface *thiz, DFBSurfacePixelFormat *ret_format ); DFBResult (*GetAccelerationMask) ( IDirectFBSurface *thiz, IDirectFBSurface *source, DFBAccelerationMask *ret_mask ); DFBResult (*GetPalette) ( IDirectFBSurface *thiz, IDirectFBPalette **ret_interface ); DFBResult (*SetPalette) ( IDirectFBSurface *thiz, IDirectFBPalette *palette ); DFBResult (*SetAlphaRamp) ( IDirectFBSurface *thiz, u8 a0, u8 a1, u8 a2, u8 a3 ); DFBResult (*Lock) ( IDirectFBSurface *thiz, DFBSurfaceLockFlags flags, void **ret_ptr, int *ret_pitch ); DFBResult (*GetFramebufferOffset) ( IDirectFBSurface *thiz, int *offset ); DFBResult (*Unlock) ( IDirectFBSurface *thiz ); DFBResult (*Flip) ( IDirectFBSurface *thiz, const DFBRegion *region, DFBSurfaceFlipFlags flags ); DFBResult (*SetField) ( IDirectFBSurface *thiz, int field ); DFBResult (*Clear) ( IDirectFBSurface *thiz, u8 r, u8 g, u8 b, u8 a ); DFBResult (*SetClip) ( IDirectFBSurface *thiz, const DFBRegion *clip ); DFBResult (*GetClip) ( IDirectFBSurface *thiz, DFBRegion *ret_clip ); DFBResult (*SetColor) ( IDirectFBSurface *thiz, u8 r, u8 g, u8 b, u8 a ); DFBResult (*SetColorIndex) ( IDirectFBSurface *thiz, unsigned int index ); DFBResult (*SetSrcBlendFunction) ( IDirectFBSurface *thiz, DFBSurfaceBlendFunction function ); DFBResult (*SetDstBlendFunction) ( IDirectFBSurface *thiz, DFBSurfaceBlendFunction function ); DFBResult (*SetPorterDuff) ( IDirectFBSurface *thiz, DFBSurfacePorterDuffRule rule ); DFBResult (*SetSrcColorKey) ( IDirectFBSurface *thiz, u8 r, u8 g, u8 b ); DFBResult (*SetSrcColorKeyIndex) ( IDirectFBSurface *thiz, unsigned int index ); DFBResult (*SetDstColorKey) ( IDirectFBSurface *thiz, u8 r, u8 g, u8 b ); DFBResult (*SetDstColorKeyIndex) ( IDirectFBSurface *thiz, unsigned int index ); DFBResult (*SetBlittingFlags) ( IDirectFBSurface *thiz, DFBSurfaceBlittingFlags flags ); DFBResult (*Blit) ( IDirectFBSurface *thiz, IDirectFBSurface *source, const DFBRectangle *source_rect, int x, int y ); DFBResult (*TileBlit) ( IDirectFBSurface *thiz, IDirectFBSurface *source, const DFBRectangle *source_rect, int x, int y ); DFBResult (*BatchBlit) ( IDirectFBSurface *thiz, IDirectFBSurface *source, const DFBRectangle *source_rects, const DFBPoint *dest_points, int num ); DFBResult (*StretchBlit) ( IDirectFBSurface *thiz, IDirectFBSurface *source, const DFBRectangle *source_rect, const DFBRectangle *destination_rect ); DFBResult (*TextureTriangles) ( IDirectFBSurface *thiz, IDirectFBSurface *texture, const DFBVertex *vertices, const int *indices, int num, DFBTriangleFormation formation ); DFBResult (*SetDrawingFlags) ( IDirectFBSurface *thiz, DFBSurfaceDrawingFlags flags ); DFBResult (*FillRectangle) ( IDirectFBSurface *thiz, int x, int y, int w, int h ); DFBResult (*DrawRectangle) ( IDirectFBSurface *thiz, int x, int y, int w, int h ); DFBResult (*DrawLine) ( IDirectFBSurface *thiz, int x1, int y1, int x2, int y2 ); DFBResult (*DrawLines) ( IDirectFBSurface *thiz, const DFBRegion *lines, unsigned int num_lines ); DFBResult (*FillTriangle) ( IDirectFBSurface *thiz, int x1, int y1, int x2, int y2, int x3, int y3 ); DFBResult (*FillRectangles) ( IDirectFBSurface *thiz, const DFBRectangle *rects, unsigned int num ); DFBResult (*FillSpans) ( IDirectFBSurface *thiz, int y, const DFBSpan *spans, unsigned int num ); DFBResult (*SetFont) ( IDirectFBSurface *thiz, IDirectFBFont *font ); DFBResult (*GetFont) ( IDirectFBSurface *thiz, IDirectFBFont **ret_font ); DFBResult (*DrawString) ( IDirectFBSurface *thiz, const char *text, int bytes, int x, int y, DFBSurfaceTextFlags flags ); DFBResult (*DrawGlyph) ( IDirectFBSurface *thiz, unsigned int character, int x, int y, DFBSurfaceTextFlags flags ); DFBResult (*SetEncoding) ( IDirectFBSurface *thiz, DFBTextEncodingID encoding ); DFBResult (*GetSubSurface) ( IDirectFBSurface *thiz, const DFBRectangle *rect, IDirectFBSurface **ret_interface ); DFBResult (*GetGL) ( IDirectFBSurface *thiz, IDirectFBGL **ret_interface ); DFBResult (*Dump) ( IDirectFBSurface *thiz, const char *directory, const char *prefix ); DFBResult (*DisableAcceleration) ( IDirectFBSurface *thiz, DFBAccelerationMask mask ); DFBResult (*ReleaseSource) ( IDirectFBSurface *thiz ); DFBResult (*SetIndexTranslation) ( IDirectFBSurface *thiz, const int *indices, int num_indices ); };
# 3833 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFBPalette { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFBPalette *thiz ); DFBResult (*Release)( IDirectFBPalette *thiz ); DFBResult (*GetCapabilities) ( IDirectFBPalette *thiz, DFBPaletteCapabilities *ret_caps ); DFBResult (*GetSize) ( IDirectFBPalette *thiz, unsigned int *ret_size ); DFBResult (*SetEntries) ( IDirectFBPalette *thiz, const DFBColor *entries, unsigned int num_entries, unsigned int offset ); DFBResult (*GetEntries) ( IDirectFBPalette *thiz, DFBColor *ret_entries, unsigned int num_entries, unsigned int offset ); DFBResult (*FindBestMatch) ( IDirectFBPalette *thiz, u8 r, u8 g, u8 b, u8 a, unsigned int *ret_index ); DFBResult (*CreateCopy) ( IDirectFBPalette *thiz, IDirectFBPalette **ret_interface ); };
# 3912 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DIKS_UP = 0x00000000,
     DIKS_DOWN = 0x00000001
} DFBInputDeviceKeyState;




typedef enum {
     DIBS_UP = 0x00000000,
     DIBS_DOWN = 0x00000001
} DFBInputDeviceButtonState;




typedef enum {
     DIBM_LEFT = 0x00000001,
     DIBM_RIGHT = 0x00000002,
     DIBM_MIDDLE = 0x00000004
} DFBInputDeviceButtonMask;




typedef enum {
     DIMM_SHIFT = (1 << DIMKI_SHIFT),
     DIMM_CONTROL = (1 << DIMKI_CONTROL),
     DIMM_ALT = (1 << DIMKI_ALT),
     DIMM_ALTGR = (1 << DIMKI_ALTGR),
     DIMM_META = (1 << DIMKI_META),
     DIMM_SUPER = (1 << DIMKI_SUPER),
     DIMM_HYPER = (1 << DIMKI_HYPER)
} DFBInputDeviceModifierMask;
# 3955 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFBInputDevice { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFBInputDevice *thiz ); DFBResult (*Release)( IDirectFBInputDevice *thiz ); DFBResult (*GetID) ( IDirectFBInputDevice *thiz, DFBInputDeviceID *ret_device_id ); DFBResult (*GetDescription) ( IDirectFBInputDevice *thiz, DFBInputDeviceDescription *ret_desc ); DFBResult (*GetKeymapEntry) ( IDirectFBInputDevice *thiz, int keycode, DFBInputDeviceKeymapEntry *ret_entry ); DFBResult (*CreateEventBuffer) ( IDirectFBInputDevice *thiz, IDirectFBEventBuffer **ret_buffer ); DFBResult (*AttachEventBuffer) ( IDirectFBInputDevice *thiz, IDirectFBEventBuffer *buffer ); DFBResult (*DetachEventBuffer) ( IDirectFBInputDevice *thiz, IDirectFBEventBuffer *buffer ); DFBResult (*GetKeyState) ( IDirectFBInputDevice *thiz, DFBInputDeviceKeyIdentifier key_id, DFBInputDeviceKeyState *ret_state ); DFBResult (*GetModifiers) ( IDirectFBInputDevice *thiz, DFBInputDeviceModifierMask *ret_modifiers ); DFBResult (*GetLockState) ( IDirectFBInputDevice *thiz, DFBInputDeviceLockState *ret_locks ); DFBResult (*GetButtons) ( IDirectFBInputDevice *thiz, DFBInputDeviceButtonMask *ret_buttons ); DFBResult (*GetButtonState) ( IDirectFBInputDevice *thiz, DFBInputDeviceButtonIdentifier button, DFBInputDeviceButtonState *ret_state ); DFBResult (*GetAxis) ( IDirectFBInputDevice *thiz, DFBInputDeviceAxisIdentifier axis, int *ret_pos ); DFBResult (*GetXY) ( IDirectFBInputDevice *thiz, int *ret_x, int *ret_y ); };
# 4092 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DFEC_NONE = 0x00,
     DFEC_INPUT = 0x01,
     DFEC_WINDOW = 0x02,
     DFEC_USER = 0x03,
     DFEC_UNIVERSAL = 0x04,
     DFEC_VIDEOPROVIDER = 0x05
} DFBEventClass;




typedef enum {
     DIET_UNKNOWN = 0,
     DIET_KEYPRESS,
     DIET_KEYRELEASE,
     DIET_BUTTONPRESS,
     DIET_BUTTONRELEASE,
     DIET_AXISMOTION
} DFBInputEventType;




typedef enum {
     DIEF_NONE = 0x000,
     DIEF_TIMESTAMP = 0x001,
     DIEF_AXISABS = 0x002,
     DIEF_AXISREL = 0x004,

     DIEF_KEYCODE = 0x008,

     DIEF_KEYID = 0x010,

     DIEF_KEYSYMBOL = 0x020,

     DIEF_MODIFIERS = 0x040,

     DIEF_LOCKS = 0x080,

     DIEF_BUTTONS = 0x100,

     DIEF_GLOBAL = 0x200,





     DIEF_REPEAT = 0x400,
     DIEF_FOLLOW = 0x800
} DFBInputEventFlags;




typedef struct {
     DFBEventClass clazz;

     DFBInputEventType type;
     DFBInputDeviceID device_id;
     DFBInputEventFlags flags;



     struct timeval timestamp;


     int key_code;



     DFBInputDeviceKeyIdentifier key_id;

     DFBInputDeviceKeySymbol key_symbol;



     DFBInputDeviceModifierMask modifiers;

     DFBInputDeviceLockState locks;



     DFBInputDeviceButtonIdentifier button;

     DFBInputDeviceButtonMask buttons;



     DFBInputDeviceAxisIdentifier axis;


     int axisabs;

     int axisrel;

} DFBInputEvent;




typedef enum {
     DWET_NONE = 0x00000000,

     DWET_POSITION = 0x00000001,


     DWET_SIZE = 0x00000002,


     DWET_CLOSE = 0x00000004,

     DWET_DESTROYED = 0x00000008,


     DWET_GOTFOCUS = 0x00000010,
     DWET_LOSTFOCUS = 0x00000020,

     DWET_KEYDOWN = 0x00000100,

     DWET_KEYUP = 0x00000200,


     DWET_BUTTONDOWN = 0x00010000,

     DWET_BUTTONUP = 0x00020000,

     DWET_MOTION = 0x00040000,

     DWET_ENTER = 0x00080000,

     DWET_LEAVE = 0x00100000,

     DWET_WHEEL = 0x00200000,


     DWET_POSITION_SIZE = DWET_POSITION | DWET_SIZE,



     DWET_ALL = 0x003F033F
} DFBWindowEventType;




typedef struct {
     DFBEventClass clazz;

     DFBWindowEventType type;
     DFBWindowID window_id;



     int x;


     int y;





     int cx;
     int cy;


     int step;


     int w;
     int h;


     int key_code;



     DFBInputDeviceKeyIdentifier key_id;

     DFBInputDeviceKeySymbol key_symbol;


     DFBInputDeviceModifierMask modifiers;
     DFBInputDeviceLockState locks;


     DFBInputDeviceButtonIdentifier button;


     DFBInputDeviceButtonMask buttons;


     struct timeval timestamp;
} DFBWindowEvent;




typedef enum {
     DVPET_NONE = 0x00000000,
     DVPET_STARTED = 0x00000001,
     DVPET_STOPPED = 0x00000002,
     DVPET_SPEEDCHANGE = 0x00000004,
     DVPET_STREAMCHANGE = 0x00000008,
     DVPET_FATALERROR = 0x00000010,
     DVPET_PTSPROCESSED = 0x00000020,
     DVPET_DATAEXHAUSTED = 0x00000040,
     DVPET_VIDEOACTION = 0x00000080,
     DVPET_DATALOW = 0x00000100,
     DVPET_DATAHIGH = 0x00000200,
     DVPET_BUFFERTIMELOW = 0x00000400,
     DVPET_BUFFERTIMEHIGH = 0x00000800,
     DVPET_ALL = 0x00000FFF

} DFBVideoProviderEventType;




typedef enum {
     DVPEDST_UNKNOWN = 0x00000000,
     DVPEDST_AUDIO = 0x00000001,
     DVPEDST_VIDEO = 0x00000002,
     DVPEDST_DATA = 0x00000004,
     DVPEDST_ALL = 0x00000007,

} DFBVideoProviderEventDataSubType;




typedef struct {
     DFBEventClass clazz;

     DFBVideoProviderEventType type;
     DFBVideoProviderEventDataSubType data_type;
     int data[4];


} DFBVideoProviderEvent;




typedef struct {
     DFBEventClass clazz;

     unsigned int type;
     void *data;
} DFBUserEvent;




typedef struct {
     DFBEventClass clazz;
     unsigned int size;




} DFBUniversalEvent;




typedef union {
     DFBEventClass clazz;
     DFBInputEvent input;
     DFBWindowEvent window;
     DFBUserEvent user;
     DFBUniversalEvent universal;
     DFBVideoProviderEvent videoprovider;
} DFBEvent;






typedef struct {
     unsigned int num_events;

     unsigned int DFEC_INPUT;
     unsigned int DFEC_WINDOW;
     unsigned int DFEC_USER;
     unsigned int DFEC_UNIVERSAL;
     unsigned int DFEC_VIDEOPROVIDER;

     unsigned int DIET_KEYPRESS;
     unsigned int DIET_KEYRELEASE;
     unsigned int DIET_BUTTONPRESS;
     unsigned int DIET_BUTTONRELEASE;
     unsigned int DIET_AXISMOTION;

     unsigned int DWET_POSITION;
     unsigned int DWET_SIZE;
     unsigned int DWET_CLOSE;
     unsigned int DWET_DESTROYED;
     unsigned int DWET_GOTFOCUS;
     unsigned int DWET_LOSTFOCUS;
     unsigned int DWET_KEYDOWN;
     unsigned int DWET_KEYUP;
     unsigned int DWET_BUTTONDOWN;
     unsigned int DWET_BUTTONUP;
     unsigned int DWET_MOTION;
     unsigned int DWET_ENTER;
     unsigned int DWET_LEAVE;
     unsigned int DWET_WHEEL;
     unsigned int DWET_POSITION_SIZE;

     unsigned int DVPET_STARTED;
     unsigned int DVPET_STOPPED;
     unsigned int DVPET_SPEEDCHANGE;
     unsigned int DVPET_STREAMCHANGE;
     unsigned int DVPET_FATALERROR;
     unsigned int DVPET_PTSPROCESSED;
     unsigned int DVPET_DATAEXHAUSTED;
     unsigned int DVPET_DATALOW;
     unsigned int DVPET_DATAHIGH;
     unsigned int DVPET_BUFFERTIMELOW;
     unsigned int DVPET_BUFFERTIMEHIGH;
} DFBEventBufferStats;
# 4425 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFBEventBuffer { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFBEventBuffer *thiz ); DFBResult (*Release)( IDirectFBEventBuffer *thiz ); DFBResult (*Reset) ( IDirectFBEventBuffer *thiz ); DFBResult (*WaitForEvent) ( IDirectFBEventBuffer *thiz ); DFBResult (*WaitForEventWithTimeout) ( IDirectFBEventBuffer *thiz, unsigned int seconds, unsigned int milli_seconds ); DFBResult (*GetEvent) ( IDirectFBEventBuffer *thiz, DFBEvent *ret_event ); DFBResult (*PeekEvent) ( IDirectFBEventBuffer *thiz, DFBEvent *ret_event ); DFBResult (*HasEvent) ( IDirectFBEventBuffer *thiz ); DFBResult (*PostEvent) ( IDirectFBEventBuffer *thiz, const DFBEvent *event ); DFBResult (*WakeUp) ( IDirectFBEventBuffer *thiz ); DFBResult (*CreateFileDescriptor) ( IDirectFBEventBuffer *thiz, int *ret_fd ); DFBResult (*EnableStatistics) ( IDirectFBEventBuffer *thiz, DFBBoolean enable ); DFBResult (*GetStatistics) ( IDirectFBEventBuffer *thiz, DFBEventBufferStats *ret_stats ); };
# 4556 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DWOP_NONE = 0x00000000,
     DWOP_COLORKEYING = 0x00000001,
     DWOP_ALPHACHANNEL = 0x00000002,

     DWOP_OPAQUE_REGION = 0x00000004,

     DWOP_SHAPED = 0x00000008,


     DWOP_KEEP_POSITION = 0x00000010,

     DWOP_KEEP_SIZE = 0x00000020,

     DWOP_KEEP_STACKING = 0x00000040,

     DWOP_GHOST = 0x00001000,


     DWOP_INDESTRUCTIBLE = 0x00002000,

     DWOP_SCALE = 0x00010000,

     DWOP_ALL = 0x0001307F
} DFBWindowOptions;




typedef enum {
     DWSC_MIDDLE = 0x00000000,

     DWSC_UPPER = 0x00000001,




     DWSC_LOWER = 0x00000002




} DFBWindowStackingClass;
# 4607 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFBWindow { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFBWindow *thiz ); DFBResult (*Release)( IDirectFBWindow *thiz ); DFBResult (*GetID) ( IDirectFBWindow *thiz, DFBWindowID *ret_window_id ); DFBResult (*GetPosition) ( IDirectFBWindow *thiz, int *ret_x, int *ret_y ); DFBResult (*GetSize) ( IDirectFBWindow *thiz, int *ret_width, int *ret_height ); DFBResult (*CreateEventBuffer) ( IDirectFBWindow *thiz, IDirectFBEventBuffer **ret_buffer ); DFBResult (*AttachEventBuffer) ( IDirectFBWindow *thiz, IDirectFBEventBuffer *buffer ); DFBResult (*DetachEventBuffer) ( IDirectFBWindow *thiz, IDirectFBEventBuffer *buffer ); DFBResult (*EnableEvents) ( IDirectFBWindow *thiz, DFBWindowEventType mask ); DFBResult (*DisableEvents) ( IDirectFBWindow *thiz, DFBWindowEventType mask ); DFBResult (*GetSurface) ( IDirectFBWindow *thiz, IDirectFBSurface **ret_surface ); DFBResult (*SetProperty) ( IDirectFBWindow *thiz, const char *key, void *value, void **ret_old_value ); DFBResult (*GetProperty) ( IDirectFBWindow *thiz, const char *key, void **ret_value ); DFBResult (*RemoveProperty) ( IDirectFBWindow *thiz, const char *key, void **ret_value ); DFBResult (*SetOptions) ( IDirectFBWindow *thiz, DFBWindowOptions options ); DFBResult (*GetOptions) ( IDirectFBWindow *thiz, DFBWindowOptions *ret_options ); DFBResult (*SetColorKey) ( IDirectFBWindow *thiz, u8 r, u8 g, u8 b ); DFBResult (*SetColorKeyIndex) ( IDirectFBWindow *thiz, unsigned int index ); DFBResult (*SetOpacity) ( IDirectFBWindow *thiz, u8 opacity ); DFBResult (*SetOpaqueRegion) ( IDirectFBWindow *thiz, int x1, int y1, int x2, int y2 ); DFBResult (*GetOpacity) ( IDirectFBWindow *thiz, u8 *ret_opacity ); DFBResult (*SetCursorShape) ( IDirectFBWindow *thiz, IDirectFBSurface *shape, int hot_x, int hot_y ); DFBResult (*RequestFocus) ( IDirectFBWindow *thiz ); DFBResult (*GrabKeyboard) ( IDirectFBWindow *thiz ); DFBResult (*UngrabKeyboard) ( IDirectFBWindow *thiz ); DFBResult (*GrabPointer) ( IDirectFBWindow *thiz ); DFBResult (*UngrabPointer) ( IDirectFBWindow *thiz ); DFBResult (*GrabKey) ( IDirectFBWindow *thiz, DFBInputDeviceKeySymbol symbol, DFBInputDeviceModifierMask modifiers ); DFBResult (*UngrabKey) ( IDirectFBWindow *thiz, DFBInputDeviceKeySymbol symbol, DFBInputDeviceModifierMask modifiers ); DFBResult (*Move) ( IDirectFBWindow *thiz, int dx, int dy ); DFBResult (*MoveTo) ( IDirectFBWindow *thiz, int x, int y ); DFBResult (*Resize) ( IDirectFBWindow *thiz, int width, int height ); DFBResult (*SetStackingClass) ( IDirectFBWindow *thiz, DFBWindowStackingClass stacking_class ); DFBResult (*Raise) ( IDirectFBWindow *thiz ); DFBResult (*Lower) ( IDirectFBWindow *thiz ); DFBResult (*RaiseToTop) ( IDirectFBWindow *thiz ); DFBResult (*LowerToBottom) ( IDirectFBWindow *thiz ); DFBResult (*PutAtop) ( IDirectFBWindow *thiz, IDirectFBWindow *lower ); DFBResult (*PutBelow) ( IDirectFBWindow *thiz, IDirectFBWindow *upper ); DFBResult (*Close) ( IDirectFBWindow *thiz ); DFBResult (*Destroy) ( IDirectFBWindow *thiz ); DFBResult (*SetBounds) ( IDirectFBWindow *thiz, int x, int y, int width, int height ); DFBResult (*ResizeSurface) ( IDirectFBWindow *thiz, int width, int height ); };
# 5036 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef DFBEnumerationResult (*DFBTextEncodingCallback) (
     DFBTextEncodingID encoding_id,
     const char *name,
     void *context
);
# 5049 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFBFont { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFBFont *thiz ); DFBResult (*Release)( IDirectFBFont *thiz ); DFBResult (*GetAscender) ( IDirectFBFont *thiz, int *ret_ascender ); DFBResult (*GetDescender) ( IDirectFBFont *thiz, int *ret_descender ); DFBResult (*GetHeight) ( IDirectFBFont *thiz, int *ret_height ); DFBResult (*GetMaxAdvance) ( IDirectFBFont *thiz, int *ret_maxadvance ); DFBResult (*GetKerning) ( IDirectFBFont *thiz, unsigned int prev, unsigned int current, int *ret_kern_x, int *ret_kern_y ); DFBResult (*GetStringWidth) ( IDirectFBFont *thiz, const char *text, int bytes, int *ret_width ); DFBResult (*GetStringExtents) ( IDirectFBFont *thiz, const char *text, int bytes, DFBRectangle *ret_logical_rect, DFBRectangle *ret_ink_rect ); DFBResult (*GetGlyphExtents) ( IDirectFBFont *thiz, unsigned int character, DFBRectangle *ret_rect, int *ret_advance ); DFBResult (*GetStringBreak) ( IDirectFBFont *thiz, const char *text, int bytes, int max_width, int *ret_width, int *ret_str_length, const char **ret_next_line ); DFBResult (*SetEncoding) ( IDirectFBFont *thiz, DFBTextEncodingID encoding ); DFBResult (*EnumEncodings) ( IDirectFBFont *thiz, DFBTextEncodingCallback callback, void *context ); DFBResult (*FindEncoding) ( IDirectFBFont *thiz, const char *name, DFBTextEncodingID *ret_encoding ); };
# 5246 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DICAPS_NONE = 0x00000000,
     DICAPS_ALPHACHANNEL = 0x00000001,

     DICAPS_COLORKEY = 0x00000002


} DFBImageCapabilities;





typedef struct {
     DFBImageCapabilities caps;

     u8 colorkey_r;
     u8 colorkey_g;
     u8 colorkey_b;
} DFBImageDescription;


typedef enum {
        DIRCR_OK,
        DIRCR_ABORT
} DIRenderCallbackResult;




typedef DIRenderCallbackResult (*DIRenderCallback)(DFBRectangle *rect, void *ctx);
# 5285 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFBImageProvider { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFBImageProvider *thiz ); DFBResult (*Release)( IDirectFBImageProvider *thiz ); DFBResult (*GetSurfaceDescription) ( IDirectFBImageProvider *thiz, DFBSurfaceDescription *ret_dsc ); DFBResult (*GetImageDescription) ( IDirectFBImageProvider *thiz, DFBImageDescription *ret_dsc ); DFBResult (*RenderTo) ( IDirectFBImageProvider *thiz, IDirectFBSurface *destination, const DFBRectangle *destination_rect ); DFBResult (*SetRenderCallback) ( IDirectFBImageProvider *thiz, DIRenderCallback callback, void *callback_data ); };
# 5351 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef enum {
     DVSCAPS_NONE = 0x00000000,
     DVSCAPS_VIDEO = 0x00000001,
     DVSCAPS_AUDIO = 0x00000002

} DFBStreamCapabilities;
# 5369 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
typedef struct {
     DFBStreamCapabilities caps;

     struct {
          char encoding[30];

          double framerate;
          double aspect;
          int bitrate;
          int afd;
          int width;
          int height;
     }
      video;

     struct {
          char encoding[30];

          int samplerate;
          int channels;
          int bitrate;
     }
      audio;

    struct {
        DFBBoolean valid;
        char LicenceFilename1[255];
        char LicenceFilename2[255];
        char MeteringFilename[255];
        double MeteringThreshold;
        DFBBoolean EnableCopyProtection;
        DFBBoolean EnableMobileCopyProtection;
        DFBBoolean DisableSecureClock;
        DFBBoolean DisableMetering;
        DFBBoolean DisableHttpDrm;
        void* EventHandle;
    }
      drm;

     char title[255];
     char author[255];
     char album[255];
     short year;
     char genre[32];
     char comment[255];
} DFBStreamDescription;




typedef enum {
     DSF_ES = 0x00000000,
     DSF_PES = 0x00000001,
} DFBStreamFormat;




typedef struct {
     DFBStreamCapabilities caps;
     struct {
          char encoding[30];

          DFBStreamFormat format;
          double maxSmoothMultiplier;



     } video;

     struct {
          char encoding[30];

          DFBStreamFormat format;
     } audio;

    struct {
        DFBBoolean valid;
        char LicenceFilename1[255];
        char LicenceFilename2[255];
        char MeteringFilename[255];
        double MeteringThreshold;
        DFBBoolean EnableCopyProtection;
        DFBBoolean EnableMobileCopyProtection;
        DFBBoolean DisableSecureClock;
        DFBBoolean DisableMetering;
        DFBBoolean DisableHttpDrm;
        void* EventHandle;
    }
      drm;
} DFBStreamAttributes;




typedef struct
{
     DFBStreamCapabilities valid;
     struct
     {
         unsigned int buffer_size;
         unsigned int minimum_level;
         unsigned int maximum_level;
         unsigned int current_level;
     } video;

     struct
     {
         unsigned int buffer_size;
         unsigned int minimum_level;
         unsigned int maximum_level;
         unsigned int current_level;
     } audio;
} DFBBufferOccupancy;




typedef struct
{
     DFBStreamCapabilities selection;
     struct
     {
         unsigned int minimum_level;
         unsigned int maximum_level;
         unsigned int minimum_time;
         unsigned int maximum_time;
     } video;

     struct
     {
         unsigned int minimum_level;
         unsigned int maximum_level;
         unsigned int minimum_time;
         unsigned int maximum_time;
     } audio;
} DFBBufferThresholds;




typedef void (*DVFrameCallback)(void *ctx);
# 5520 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFBVideoProvider { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFBVideoProvider *thiz ); DFBResult (*Release)( IDirectFBVideoProvider *thiz ); DFBResult (*GetCapabilities) ( IDirectFBVideoProvider *thiz, DFBVideoProviderCapabilities *ret_caps ); DFBResult (*GetSurfaceDescription) ( IDirectFBVideoProvider *thiz, DFBSurfaceDescription *ret_dsc ); DFBResult (*GetStreamDescription) ( IDirectFBVideoProvider *thiz, DFBStreamDescription *ret_dsc ); DFBResult (*GetBufferOccupancy) ( IDirectFBVideoProvider *thiz, DFBBufferOccupancy *ret_occ ); DFBResult (*SetBufferThresholds) ( IDirectFBVideoProvider *thiz, DFBBufferThresholds thresh ); DFBResult (*GetBufferThresholds) ( IDirectFBVideoProvider *thiz, DFBBufferThresholds *ret_thresh ); DFBResult (*SetStreamAttributes) ( IDirectFBVideoProvider *thiz, DFBStreamAttributes attr ); DFBResult (*CreateEventBuffer) ( IDirectFBVideoProvider *thiz, IDirectFBEventBuffer **ret_buffer ); DFBResult (*AttachEventBuffer) ( IDirectFBVideoProvider *thiz, IDirectFBEventBuffer *buffer ); DFBResult (*EnableEvents) ( IDirectFBVideoProvider *thiz, DFBVideoProviderEventType mask ); DFBResult (*DisableEvents) ( IDirectFBVideoProvider *thiz, DFBVideoProviderEventType mask ); DFBResult (*DetachEventBuffer) ( IDirectFBVideoProvider *thiz, IDirectFBEventBuffer *buffer ); DFBResult (*SetAudioOutputs) ( IDirectFBVideoProvider *thiz, DFBVideoProviderAudioUnits* audioUnits ); DFBResult (*GetAudioOutputs) ( IDirectFBVideoProvider *thiz, DFBVideoProviderAudioUnits* audioUnits ); DFBResult (*PlayTo) ( IDirectFBVideoProvider *thiz, IDirectFBSurface *destination, const DFBRectangle *destination_rect, DVFrameCallback callback, void *ctx ); DFBResult (*Stop) ( IDirectFBVideoProvider *thiz ); DFBResult (*GetStatus) ( IDirectFBVideoProvider *thiz, DFBVideoProviderStatus *ret_status ); DFBResult (*SeekTo) ( IDirectFBVideoProvider *thiz, double seconds ); DFBResult (*GetPos) ( IDirectFBVideoProvider *thiz, double *ret_seconds ); DFBResult (*GetLength) ( IDirectFBVideoProvider *thiz, double *ret_seconds ); DFBResult (*GetColorAdjustment) ( IDirectFBVideoProvider *thiz, DFBColorAdjustment *ret_adj ); DFBResult (*SetColorAdjustment) ( IDirectFBVideoProvider *thiz, const DFBColorAdjustment *adj ); DFBResult (*SendEvent) ( IDirectFBVideoProvider *thiz, const DFBEvent *event ); DFBResult (*SetPlaybackFlags) ( IDirectFBVideoProvider *thiz, DFBVideoProviderPlaybackFlags flags ); DFBResult (*SetSpeed) ( IDirectFBVideoProvider *thiz, double multiplier ); DFBResult (*GetSpeed) ( IDirectFBVideoProvider *thiz, double *ret_multiplier ); DFBResult (*SetVolume) ( IDirectFBVideoProvider *thiz, float level ); DFBResult (*GetVolume) ( IDirectFBVideoProvider *thiz, float *ret_level ); DFBResult (*SetAudioDelay) ( IDirectFBVideoProvider *thiz, long delay ); DFBResult (*SetBitStreamType) ( IDirectFBVideoProvider *thiz, long bit_stream_type ); };
# 5830 "/home/lucas/software/pr11/stb225/src/open/buildroot/overlay/package/directfb/overlay1.0/include/directfb.h"
struct _IDirectFBDataBuffer { void *priv; u32 magic; DFBResult (*AddRef)( IDirectFBDataBuffer *thiz ); DFBResult (*Release)( IDirectFBDataBuffer *thiz ); DFBResult (*Flush) ( IDirectFBDataBuffer *thiz ); DFBResult (*Finish) ( IDirectFBDataBuffer *thiz ); DFBResult (*SeekTo) ( IDirectFBDataBuffer *thiz, unsigned int offset ); DFBResult (*GetPosition) ( IDirectFBDataBuffer *thiz, unsigned int *ret_offset ); DFBResult (*GetLength) ( IDirectFBDataBuffer *thiz, unsigned int *ret_length ); DFBResult (*WaitForData) ( IDirectFBDataBuffer *thiz, unsigned int length ); DFBResult (*WaitForDataWithTimeout) ( IDirectFBDataBuffer *thiz, unsigned int length, unsigned int seconds, unsigned int milli_seconds ); DFBResult (*GetData) ( IDirectFBDataBuffer *thiz, unsigned int length, void *ret_data, unsigned int *ret_read ); DFBResult (*PeekData) ( IDirectFBDataBuffer *thiz, unsigned int length, int offset, void *ret_data, unsigned int *ret_read ); DFBResult (*HasData) ( IDirectFBDataBuffer *thiz ); DFBResult (*PutData) ( IDirectFBDataBuffer *thiz, const void *data, unsigned int length ); DFBResult (*CreateImageProvider) ( IDirectFBDataBuffer *thiz, IDirectFBImageProvider **interface ); DFBResult (*CreateVideoProvider) ( IDirectFBDataBuffer *thiz, IDirectFBVideoProvider **interface ); };
# 114 "app_info.h" 2
# 154 "app_info.h"
typedef enum aac_bit_stream_type
{
    AAC_BIT_STREAM_MODE_RAW=0,
    AAC_BIT_STREAM_MODE_ADTS,
    AAC_BIT_STREAM_MODE_ADIF,
    AAC_BIT_STREAM_MODE_LATM
}aac_bit_stream_type_t;



typedef enum __playbackMode {
    playback_single,
    playback_looped,
    playback_sequential,
    playback_modes
} playbackMode_t;


typedef enum __trickModeSpeed {
    speed_1_32,
    speed_1_16,
    speed_1_8,
    speed_1_4,
    speed_1_2,
    speed_1,
    speed_2,
    speed_4,
    speed_8,
    speed_16,
    speed_32,
    speed_0,
    speedStates
} trickModeSpeed_t;


typedef enum __trickModeDirection {
    direction_none,
    direction_forwards,
    direction_backwards,
    direction_states,
} trickModeDirection_t;


typedef enum __audioStatus {
   audioMute = 0,
   audioMain,
   audioStates
} audioStatus_t;


typedef enum __screenOutput {
   screenMain = 0,
   screenPip,
   screenOutputs
} screenOutput_t;


typedef enum __videoMode {
    videoMode_stretch,
    videoMode_scale,
    videoMode_custom,
    videoMode_native
} videoMode_t;


typedef enum __aspectRatio {
    aspectRatio_4x3,
    aspectRatio_16x9
} aspectRatio_t;


typedef enum __zoomFactor {
    zoomFactor_1,
    zoomFactor_2,
    zoomFactor_4,
    zoomFactor_8,
    zoomFactor_16,
} zoomFactor_t;


typedef enum __GFXBlend {
    GFXBlend_100 = 255,
    GFXBlend_90 = 230,
    GFXBlend_80 = 205,
    GFXBlend_70 = 180,
    GFXBlend_60 = 155,
    GFXBlend_50 = 120
} GFXBlend_t;


typedef enum __tunerFormat {
   inputTuner0 = 0,
   inputTuner1,
   inputTuners
} tunerFormat_t;


typedef enum __tunerStatus {
   tunerNotPresent = 0,
   tunerInactive,
   tunerDVBPip,
   tunerDVBMain,
   tunerDVBPvr2,
   tunerDVBPvr1,
   tunerStatusGaurd
} tunerStatus_t;


typedef struct __tunerInfo {
    tunerStatus_t status;
    fe_status_t fe_status;
    uint32_t ber;
    uint16_t signal_strength;
    uint16_t snr;
    uint32_t uncorrected_blocks;
} tunerInfo_t;


typedef struct __dvbInfo {
   _Bool active;
   tunerFormat_t tuner;
   uint32_t channel;
   _Bool scrambled;
   screenOutput_t output;
} dvbInfo_t;

typedef struct __trickModeInfo {
    _Bool active;
    trickModeSpeed_t speed;
    trickModeDirection_t direction;
}trickModeInfo_t;


typedef struct __ipInfo {
   _Bool active;
   _Bool endOfFile;
   playbackMode_t playbackMode;
   uint32_t streamNumber;
   uint32_t maxStreams;
   char streamUrl[16];
   char multicastUrl[16];
   uint32_t rtspPort;
   uint32_t multicastPort;
   char pStreamName[256];
   char pStreamDestIP[256];
   char pStreamPort[256];
   char pServerName[256];
} ipInfo_t;



typedef struct __pvrPlaybackInfo {
   _Bool active;
   _Bool paused;
   _Bool endOfFile;
   trickModeInfo_t trickModeInfo;
   uint32_t fileNumber;
   char directory[1024];
   char activeDirectory[1024];
   _Bool isH264;
} pvrPlaybackInfo_t;


typedef struct __pvrRecordInfo {
   _Bool active;
   tunerFormat_t tuner;
   uint32_t channel;
   char directory[1024];
} pvrRecordInfo_t;


typedef struct __pvrInfo {
   char directory[1024];
   playbackMode_t playbackMode;
   uint32_t maxFile;
   pvrPlaybackInfo_t playbackInfo[screenOutputs];
   pvrRecordInfo_t recordInfo[inputTuners];
} pvrInfo_t;


typedef struct __pictureInfo {
   int32_t skinTone;
   int32_t greenStretch;
   int32_t blueStretch;
   GFXBlend_t osdBlend;
   GFXBlend_t imageBlend;
} pictureInfo_t;


typedef struct {
    uint32_t videoPid;
    uint8_t videoType;
    uint32_t pcrPid;
    uint32_t audioPid[10];
    uint8_t audioType[10];
    uint32_t numAudioPids;
} streamPids_t;


typedef struct __mediaInfo {
   uint32_t filter;
   _Bool active;
   _Bool paused;
   char directory[1024];
   char filename[1024];
   char activeFilename[1024];
   int32_t endOfStreamCountdown;
   _Bool endOfStreamReported;
   _Bool endOfFileReported;
   _Bool endOfVideoDataReported;
   _Bool endOfAudioDataReported;
   _Bool audioPlaybackStarted;
   _Bool videoPlaybackStarted;
   _Bool bufferingData;
   int32_t bufferingIconIndex;
   int32_t currentFile;
   int32_t maxFile;
   playbackMode_t playbackMode;
   _Bool pdSupportEnabled;
   _Bool prerollEnabled;
   trickModeInfo_t trickModeInfo;
   trickModeSpeed_t maxSmoothSpeed;
} mediaInfo_t;


typedef struct __imageInfo {
   int32_t active;
   uint32_t filter;
   int32_t duration;
   int32_t transition;
   _Bool displayFilename;
   char directory[1024];
} imageInfo_t;


typedef struct __pipInfo {
   int32_t location;
} pipInfo_t;


typedef struct __soundInfo {
   _Bool muted;
   int32_t volumeLevel;
   long audioDelay;
   _Bool keyBeep;
} soundInfo_t;


typedef struct __outputInfo {
   DFBScreenOutputConfig config;
   DFBScreenOutputDescription desc;
   DFBScreenEncoderConfig encConfig[2];
   DFBScreenEncoderDescription encDesc[2];
   videoMode_t autoScale;
   aspectRatio_t aspectRatio;
   zoomFactor_t zoomFactor;
   int32_t numberOfEncoders;
   _Bool analogSlaveToDigital;
   DFBScreenEncoderMacrovision macrovisionMode;

   DFBScreenEncoderWideScreenSignaling wssMode;

   DFBScreenEncoderCopyGenerationManagementSystem cgmsMode;

} outputInfo_t;


typedef struct __loggedCommandInfo {
   _Bool loop;
   int32_t inputFile;
   int32_t outputFile;
} loggedCommandInfo_t;


typedef struct __digitInfo {
   _Bool active;
   uint32_t value;
   int32_t numDigits;
   int32_t countDown;
} digitInfo_t;


typedef struct __programInfo
{
    _Bool valid;
    _Bool found;
    _Bool displayed;
    char startDate[32];
    char startTime[32];
    char stopDate[32];
    char stopTime[32];
    char title[256];
    char description[256];
    char videoAspect[32];
    char videoResolution[32];
    char videoFrequency[32];
    char audio[32];
    char category[64];
} programInfo_t;


typedef struct __timeInfo
{
    int32_t displayed;
    int32_t year;
    int32_t month;
    int32_t day;
    int32_t hour;
    int32_t minute;
    int32_t second;
    int32_t offset_polarity;
    int32_t offset_hours;
    int32_t offset_minutes;
} timeInfo_t;


typedef struct __statsInfo
{
    _Bool displayHistogram;
    _Bool dataUsage;
    uint32_t dataRate;
    _Bool cpuUsage;
    _Bool cpuLevelAvailable[2];
    uint32_t cpuLevel[2];
    uint32_t decodeLow;
    uint32_t decodeHigh;
    uint32_t decodeAve;
    _Bool bufferUsage;
    DFBBufferOccupancy buffOccupancy;

    DFBBufferThresholds buffThresholds;
    _Bool eventStats;
} statsInfo_t;


typedef struct __controlInfo {
   audioStatus_t audioStatus;
   tunerInfo_t tunerInfo[inputTuners];
   dvbInfo_t dvbInfo[screenOutputs];
   ipInfo_t ipInfo[screenOutputs];
   pvrInfo_t pvrInfo;
   pictureInfo_t pictureInfo;
   soundInfo_t soundInfo;
   mediaInfo_t mediaInfo;
   imageInfo_t imageInfo;
   outputInfo_t outputInfo;
   pipInfo_t pipInfo;
   loggedCommandInfo_t commandInfo;
   digitInfo_t digitInfo;
   programInfo_t programInfo;
   timeInfo_t timeInfo;
   statsInfo_t statsInfo;
   char channelConfigFile[256];
   int32_t tunerDebug;
   _Bool scanActive;
   _Bool timeout;
   int32_t displayCount;
   _Bool streamerInput;
   _Bool enableWatchdog;
   _Bool ptsLocked;
   _Bool enableAudio;
   _Bool enableVideo;
   _Bool allowStreamErrors;
   _Bool restartGraphics;
   uint32_t restartResolution;
   FILE *seekFd;
   streamPids_t streamPids;
} controlInfo_t;





extern controlInfo_t appControlInfo;
# 539 "app_info.h"
extern void appInfo_init(void);
# 133 "exStbDemo.c" 2
# 1 "gfx.h" 1
# 91 "gfx.h"
# 1 "sem.h" 1
# 68 "sem.h"
typedef struct
{
 pthread_mutex_t mutex;
 pthread_cond_t condition;
 int semCount;
}sem_t, *psem_t;
# 84 "sem.h"
extern int32_t sem_get(psem_t semaphore);


extern int32_t sem_release(psem_t semaphore);


extern int32_t sem_create(psem_t* semaphore);


extern int32_t sem_destroy(psem_t semaphore);


extern int32_t event_wait(psem_t semaphore);


extern int32_t event_waitWithTimeout(psem_t semaphore, uint32_t seconds, uint32_t milli_seconds );


extern int32_t event_send(psem_t semaphore);


extern int32_t event_create(psem_t* semaphore);


extern int32_t event_destroy(psem_t semaphore);
# 92 "gfx.h" 2
# 176 "gfx.h"
typedef struct __gfxImageEntry {
    char filename[256];
    IDirectFBSurface *pImage;
    int32_t width;
    int32_t height;
    struct __gfxImageEntry *pNext;
    struct __gfxImageEntry *pPrev;
} gfxImageEntry;


typedef enum
{
    gfxLayerNotDisplayed,
    gfxLayerDisplayed
} gfxLayerDisplay_t;

typedef enum
{
    gfxMixerAnalog,
    gfxMixerHdmi
} gfxMixer_t;

typedef enum
{
    gfxEncoderHdmi,
    gfxEncoderAnalog
} gfxEncoder_t;

typedef enum
{
    gfxStreamTypesUnknown = 0,
    gfxStreamTypesMpegTS,
    gfxStreamTypesMpegPS,
    gfxStreamTypesMpeg4,
    gfxStreamTypesMP3,
    gfxStreamTypesAnalog,
    gfxStreamTypesDivx,
    gfxStreamTypesWMT,
    gfxStreamTypesH264ES
} gfxStreamTypes_t;






extern IDirectFBSurface *pgfx_frameBuffer;


extern IDirectFBSurface *pgfx_slideShowBuffer;


extern IDirectFBSurface *pgfx_LCDFrameBuffer;


extern IDirectFBFont *pgfx_font;
extern IDirectFBFont *pgfx_smallFont;
extern IDirectFBFont *pgfx_largeFont;


extern IDirectFB *pgfx_dfb;


extern int32_t gfxWidth;
extern int32_t gfxHeight;

extern int32_t gfxScaleHeight;
extern int32_t gfxScaleWidth;
extern int32_t gfxScreenHeight;
extern int32_t gfxScreenWidth;
extern _Bool gfxUseScaleParams;
extern psem_t gfxDimensionsEvent;
# 262 "gfx.h"
extern IDirectFBSurface * gfx_decodeImage(char* filename, int32_t width, int32_t height);
# 277 "gfx.h"
extern void gfx_startVideoProvider(char* videoSource, uint32_t videoLayer, _Bool force, char* video_mode, char* audio_mode, char* sync_mode, DFBStreamAttributes *pStreamAttr);
# 288 "gfx.h"
extern void gfx_stopVideoProvider(uint32_t videoLayer, _Bool force, int32_t hideLayer);
# 299 "gfx.h"
extern DFBResult gfx_getVideoProviderStatus(uint32_t videoLayer, DFBVideoProviderStatus *ret_status);
# 310 "gfx.h"
extern void gfx_clearSurface(IDirectFBSurface *pSurface, int32_t width, int32_t height);
# 327 "gfx.h"
extern void gfx_drawRectangle(IDirectFBSurface *pSurface,
                              uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                              int32_t x, int32_t y,
                              int32_t width, int32_t height);
# 349 "gfx.h"
extern void gfx_drawText(IDirectFBSurface *pSurface, IDirectFBFont *pgfx_Font,
                         uint8_t r, uint8_t g, uint8_t b , uint8_t a,
                         int32_t x, int32_t y, const char *pText, int32_t drawBox, int32_t shadow);
# 361 "gfx.h"
extern int32_t gfx_getTextWidth(IDirectFBFont *pFont, char* pText);
# 370 "gfx.h"
extern void gfx_flipSurface(IDirectFBSurface *pSurface);
# 380 "gfx.h"
extern void gfx_init(int32_t argc, char* argv[]);






extern void gfx_terminate(void);
# 396 "gfx.h"
extern long gfx_getVideoProviderPosition(uint32_t videoLayer);
# 405 "gfx.h"
extern long gfx_getVideoProviderLength(uint32_t videoLayer);
# 414 "gfx.h"
extern void gfx_setVideoProviderPosition(uint32_t videoLayer, long position);
# 423 "gfx.h"
extern gfxStreamTypes_t gfx_getVideoProviderStreamType(uint32_t videoLayer);
# 438 "gfx.h"
extern void gfx_getVideoProviderBufferOccupancy(
    uint32_t videoLayer,
    uint32_t *pVideoSize,
    uint32_t *pVideoOccupancy,
    uint32_t *pAudioSize,
    uint32_t *pAudioOccupancy);
# 455 "gfx.h"
extern void gfx_setAudioDelay(uint32_t videoLayer, int32_t delay);
# 467 "gfx.h"
extern void gfx_setBitStreamType(uint32_t videoLayer, aac_bit_stream_type_t bstype);
# 479 "gfx.h"
extern void gfx_setupTrickMode(uint32_t videoLayer,
                               trickModeDirection_t direction,
                               trickModeSpeed_t speed);
# 491 "gfx.h"
extern _Bool gfx_isTrickModeSupported(uint32_t videoLayer);
# 502 "gfx.h"
extern void gfx_setScreenTVStandard(uint32_t encoder, DFBScreenEncoderTVStandards tvStandard, _Bool formatChange);
# 515 "gfx.h"
extern void gfx_setScreenResolution(uint32_t encoder, DFBScreenOutputResolution resolution, DFBScreenEncoderFrequency frequency, DFBScreenEncoderScanMode scanmode, _Bool formatChange);
# 525 "gfx.h"
extern void gfx_setEncoderInfo(gfxEncoder_t encoder, double gamma);
# 536 "gfx.h"
extern void gfx_setMixerInfo(gfxMixer_t mixer, uint32_t backgroundColour, gfxLayerDisplay_t *displayInfo);
# 547 "gfx.h"
extern void gfx_getMixerInfo(gfxMixer_t mixer, uint32_t * pBackgroundColour, gfxLayerDisplay_t *displayInfo);
# 557 "gfx.h"
extern void gfx_setLayerDepth(uint32_t whichLayer, uint32_t videoDepth);
# 567 "gfx.h"
extern void gfx_flickerFilterLayer(int32_t layer, _Bool enable);
# 579 "gfx.h"
extern void gfx_blendLayer(_Bool fadeOut, _Bool mainLayer, int32_t steps, int32_t maxValue);
# 593 "gfx.h"
extern void gfx_blendVideoLayer(uint32_t videoLayer, _Bool fadeOut, int32_t steps, int32_t maxValue);
# 603 "gfx.h"
extern void gfx_sendEvent(int32_t videoLayer, DFBEvent *event);






extern void gfx_getBufferInformation(void);
# 619 "gfx.h"
extern _Bool gfx_eventIsMonitored(uint32_t event);
# 628 "gfx.h"
extern uint32_t gfx_getEventCount(uint32_t event);
# 637 "gfx.h"
extern const char* gfx_getEventName(uint32_t event);
# 647 "gfx.h"
extern void gfx_switchAudioEncoding( uint32_t videoLayer, char* audio_encoding );
# 657 "gfx.h"
extern void gfx_setMaxSmoothSpeed( uint32_t videoLayer, trickModeSpeed_t speed);
# 134 "exStbDemo.c" 2
# 1 "menu_infra.h" 1
# 82 "menu_infra.h"
typedef enum __menuEntryType {
   menuEntryImage = 0,
   menuEntryText,
   menuEntryHeading,
   menuEntryTypes
} menuEntryType_t;

typedef int32_t menuActionFunction_t(void*);
typedef long sliderGetFunction_t(void);
typedef void sliderSetFunction_t(long);

struct __menuContainer;
typedef int32_t menuHighlightFunction_t(struct __menuContainer*);

typedef struct __menuEntry {
    menuEntryType_t type;
    char *icon;
    uint32_t iconAllocSize;
    char *info;
    uint32_t infoAllocSize;
    menuActionFunction_t *pAction;
    void *pArg;
} menuEntry_t;

typedef struct __menuContainer {
    struct __menuContainer *prev;
    int32_t highlight;
    int32_t selected;
    int32_t numEntries;
    int32_t firstShownIndex;
    int32_t lastShownIndex;
    _Bool menuOverflow;
    menuEntry_t menuEntry[(512)];
    menuHighlightFunction_t *pHighlight;
} menuContainer_t;

typedef struct __sliderContainer {
    struct __menuContainer *prev;
    char *name;
    uint32_t nameAllocSize;
    long minValue;
    long maxValue;
    int32_t divisions;
    int32_t width;
    int32_t posX;
    int32_t posY;
    sliderGetFunction_t *pGetFunction;
    sliderSetFunction_t *pSetFunction;
} sliderContainer_t;





extern menuContainer_t *pCurrentMenu;
extern menuContainer_t *pLastDisplayedMenu;
extern sliderContainer_t *pCurrentSlider;
# 160 "menu_infra.h"
extern void menuInfra_setEntry(menuContainer_t *pMenu, int32_t position,
                               menuEntryType_t entryType, char const * description,
                               menuActionFunction_t *pAction, void* pArg);
# 174 "menu_infra.h"
extern void menuInfra_setEntryIcon(menuContainer_t *pMenu, int32_t which, char* icon);
# 184 "menu_infra.h"
extern void menuInfra_setHighlight(menuContainer_t *pMenu, int32_t which);
# 194 "menu_infra.h"
extern void menuInfra_setSelected(menuContainer_t *pMenu, int32_t which);
# 203 "menu_infra.h"
extern int32_t menuInfra_display(void* pArg);






extern void menuInfra_updateOSD(void);
# 219 "menu_infra.h"
extern int32_t menuInfra_sliderDisplay(void* pArg);
# 229 "menu_infra.h"
extern void menuInfra_setSliderName(sliderContainer_t *pSlider, char* name);
# 239 "menu_infra.h"
extern void menuInfra_setSliderWidth(sliderContainer_t *pSlider, int32_t width);
# 250 "menu_infra.h"
extern void menuInfra_setSliderPosition(sliderContainer_t *pSlider, int32_t posX, int32_t posY);
# 261 "menu_infra.h"
extern void menuInfra_setSliderRange(sliderContainer_t *pSlider, long min, long max);
# 272 "menu_infra.h"
extern void menuInfra_setSliderAccessFunctions(sliderContainer_t *pSlider, sliderGetFunction_t *pGetFunction, sliderSetFunction_t *pSetFunction);
# 281 "menu_infra.h"
extern void menuInfra_sliderIncrement(sliderContainer_t *pSlider);
# 290 "menu_infra.h"
extern void menuInfra_sliderDecrement(sliderContainer_t *pSlider);
# 299 "menu_infra.h"
extern void menuInfra_sliderClose(sliderContainer_t *pSlider);
# 308 "menu_infra.h"
extern void menuInfra_freeMenuEntries(menuContainer_t *pMenu);
# 317 "menu_infra.h"
extern void menuInfra_freeSliderEntries(sliderContainer_t *pSlider);
# 135 "exStbDemo.c" 2
# 1 "menu_app.h" 1
# 74 "menu_app.h"
extern menuContainer_t topLevelMenu;
# 85 "menu_app.h"
extern void menuApp_buildInitial(void);






extern void menuApp_terminate(void);






extern void menuApp_displayMain(void);






extern void menuApp_displayPrevious(void);







extern void menuApp_doAction(void);






extern void menuApp_highlightUp(void);






extern void menuApp_highlightDown(void);







extern void menuApp_highlightPageUp(void);







extern void menuApp_highlightPageDown(void);






extern void menuApp_channelUp(void);






extern void menuApp_channelDown(void);






extern void menuApp_channelChange(uint32_t which);
# 136 "exStbDemo.c" 2
# 1 "dvb.h" 1
# 73 "dvb.h"
typedef enum DvbMode_enum {
    DvbMode_Watch,
    DvbMode_Record,
    DvbMode_Play
} DvbMode_t;


typedef void dvb_displayFunctionDef(long , uint32_t, tunerFormat_t);

typedef struct dvb_filePosition {
    int32_t index;
    int32_t offset;
}DvbFilePosition_t;

typedef struct __dvb_playParam {
    int32_t audioPid;
    int32_t videoPid;
    int32_t pcrPid;
    _Bool *pEndOfFile;
    DvbFilePosition_t position;
}DvbPlayParam_t;

typedef struct __dvb_liveParam {
    uint32_t channelIndex;
}DvbLiveParam_t;

typedef struct __dvb_param {
    DvbMode_t mode;
    int32_t vmsp;
    char *directory;
    union
    {
        DvbLiveParam_t liveParam;
        DvbPlayParam_t playParam;
    } param;
}DvbParam_t;
# 128 "dvb.h"
extern void dvb_getTuner_freqs( long * low_freq, long * high_freq, long * freq_step, tunerFormat_t tuner);
# 138 "dvb.h"
extern void dvb_startDVB(DvbParam_t *pParam);
# 149 "dvb.h"
extern void dvb_stopDVB(int32_t vmsp, int32_t reset);







extern uint32_t dvb_getNumberOfChannels(void);
# 167 "dvb.h"
extern char* dvb_getChannelName(uint32_t which);
# 177 "dvb.h"
extern _Bool dvb_validChannel(uint32_t which);
# 190 "dvb.h"
extern int32_t dvb_getPIDs(uint32_t which, int32_t* pVideo, int32_t* pAudio, int32_t* pPcr);
# 199 "dvb.h"
extern int32_t dvb_hasAC3Audio(uint32_t which);







extern void dvb_init(void);







extern void dvb_terminate(void);
# 225 "dvb.h"
extern void dvb_serviceScan(tunerFormat_t tuner, dvb_displayFunctionDef* pFunction);
# 235 "dvb.h"
extern void dvb_getPvrLength(int32_t which, DvbFilePosition_t *pPosition);
# 246 "dvb.h"
extern void dvb_getPvrPosition(int32_t which, DvbFilePosition_t *pPosition);
# 255 "dvb.h"
extern int32_t dvb_getServiceID(uint32_t which);
# 264 "dvb.h"
extern uint32_t dvb_getChannelNumber(uint32_t which);

extern uint32_t dvb_getChannelRealNumber(uint32_t which);
# 275 "dvb.h"
extern uint32_t dvb_getChannelPosition(uint32_t which);
# 284 "dvb.h"
extern char* dvb_getChannelVideoFormat(uint32_t which);
# 294 "dvb.h"
extern char* dvb_getChannelAudioFormat(uint32_t which);
# 304 "dvb.h"
extern _Bool dvb_getScrambled(uint32_t which);
# 313 "dvb.h"
extern int32_t dvb_getPvrRate(int32_t which);
# 137 "exStbDemo.c" 2
# 1 "exStbDemo.h" 1
# 61 "exStbDemo.h"
# 1 "/home/lucas/software/pr11/stb225/src/comps/phStbDbg/inc/phStbDbg.h" 1
# 80 "/home/lucas/software/pr11/stb225/src/comps/phStbDbg/inc/phStbDbg.h"
# 1 "/home/lucas/software/pr11/stb225/src/intfs/IphStbCommon/inc/phStbCommon.h" 1
# 406 "/home/lucas/software/pr11/stb225/src/intfs/IphStbCommon/inc/phStbCommon.h"
typedef union phStbColour3
{
    UBits32 u32;
# 429 "/home/lucas/software/pr11/stb225/src/intfs/IphStbCommon/inc/phStbCommon.h"
    struct {
        UBits32 blue : 8;
        UBits32 green : 8;
        UBits32 red : 8;
        UBits32 : 8;
    } rgb;
    struct {
        UBits32 v : 8;
        UBits32 u : 8;
        UBits32 y : 8;
        UBits32 : 8;
    } yuv;
    struct {
        UBits32 l : 8;
        UBits32 m : 8;
        UBits32 u : 8;
        UBits32 : 8;
    } uml;

} phStbColour3_t;


typedef union phStbColour4
{
    UBits32 u32;
# 474 "/home/lucas/software/pr11/stb225/src/intfs/IphStbCommon/inc/phStbCommon.h"
    struct {
        UBits32 blue : 8;
        UBits32 green : 8;
        UBits32 red : 8;
        UBits32 alpha : 8;
    } argb;
    struct {
        UBits32 v : 8;
        UBits32 u : 8;
        UBits32 y : 8;
        UBits32 alpha : 8;
    } ayuv;
    struct {
        UBits32 l : 8;
        UBits32 m : 8;
        UBits32 u : 8;
        UBits32 alpha : 8;
    } auml;

} phStbColour4_t;




typedef struct phStb8BitColourLut
{
    phStbColour4_t values[255];
} phStb8BitColourLut_t;

typedef struct phStbClock
{
    UInt32 clockHi;
    UInt32 clockLo;
    UInt32 clockExtn;
} phStbClock_t;
# 81 "/home/lucas/software/pr11/stb225/src/comps/phStbDbg/inc/phStbDbg.h" 2
# 196 "/home/lucas/software/pr11/stb225/src/comps/phStbDbg/inc/phStbDbg.h"
extern Void tmDbg_Init(Void);
# 458 "/home/lucas/software/pr11/stb225/src/comps/phStbDbg/inc/phStbDbg.h"
extern Void tmDbg_AssertError(
    const Char *assertion,
    const Char *file,
    UInt line,
    const Char *msg);




extern Char *tmDbg_StrAssertFormat(Char *fmt, ...);
# 62 "exStbDemo.h" 2
# 116 "exStbDemo.h"
extern int32_t exStbDemo_fileExists(char* filename);
# 126 "exStbDemo.h"
extern void exStbDemo_threadRename(char* name);
# 138 "exStbDemo.c" 2
# 1 "slideshow.h" 1
# 73 "slideshow.h"
extern int32_t slideShow_durationValue[(8)];
# 85 "slideshow.h"
extern int32_t slideShow_init(void);







extern void slideShow_terminate(void);
# 139 "exStbDemo.c" 2
# 1 "sound.h" 1
# 59 "sound.h"
# 1 "/usr/include/alsa/asoundlib.h" 1 3
# 36 "/usr/include/alsa/asoundlib.h" 3
# 1 "/usr/include/assert.h" 1 3
# 66 "/usr/include/assert.h" 3



extern void __assert_fail (__const char *__assertion, __const char *__file,
      unsigned int __line, __const char *__function)
     __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));


extern void __assert_perror_fail (int __errnum, __const char *__file,
      unsigned int __line,
      __const char *__function)
     __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));




extern void __assert (const char *__assertion, const char *__file, int __line)
     __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));



# 37 "/usr/include/alsa/asoundlib.h" 2 3

# 1 "/usr/include/sys/poll.h" 1 3
# 26 "/usr/include/sys/poll.h" 3
# 1 "/usr/include/bits/poll.h" 1 3
# 27 "/usr/include/sys/poll.h" 2 3
# 37 "/usr/include/sys/poll.h" 3
typedef unsigned long int nfds_t;


struct pollfd
  {
    int fd;
    short int events;
    short int revents;
  };



# 58 "/usr/include/sys/poll.h" 3
extern int poll (struct pollfd *__fds, nfds_t __nfds, int __timeout);
# 72 "/usr/include/sys/poll.h" 3

# 39 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/errno.h" 1 3
# 40 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stdarg.h" 1 3 4
# 105 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stdarg.h" 3 4
typedef __gnuc_va_list va_list;
# 41 "/usr/include/alsa/asoundlib.h" 2 3

# 1 "/usr/include/alsa/asoundef.h" 1 3
# 43 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/version.h" 1 3
# 44 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/global.h" 1 3
# 47 "/usr/include/alsa/global.h" 3
const char *snd_asoundlib_version(void);
# 66 "/usr/include/alsa/global.h" 3
struct snd_dlsym_link {
 struct snd_dlsym_link *next;
 const char *dlsym_name;
 const void *dlsym_ptr;
};

extern struct snd_dlsym_link *snd_dlsym_start;
# 100 "/usr/include/alsa/global.h" 3
void *snd_dlopen(const char *file, int mode);
void *snd_dlsym(void *handle, const char *name, const char *version);
int snd_dlclose(void *handle);
# 114 "/usr/include/alsa/global.h" 3
typedef struct _snd_async_handler snd_async_handler_t;






typedef void (*snd_async_callback_t)(snd_async_handler_t *handler);

int snd_async_add_handler(snd_async_handler_t **handler, int fd,
     snd_async_callback_t callback, void *private_data);
int snd_async_del_handler(snd_async_handler_t *handler);
int snd_async_handler_get_fd(snd_async_handler_t *handler);
int snd_async_handler_get_signo(snd_async_handler_t *handler);
void *snd_async_handler_get_callback_private(snd_async_handler_t *handler);

struct snd_shm_area *snd_shm_area_create(int shmid, void *ptr);
struct snd_shm_area *snd_shm_area_share(struct snd_shm_area *area);
int snd_shm_area_destroy(struct snd_shm_area *area);

int snd_user_file(const char *file, char **result);
# 149 "/usr/include/alsa/global.h" 3
typedef struct timeval snd_timestamp_t;

typedef struct timespec snd_htimestamp_t;
# 45 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/input.h" 1 3
# 54 "/usr/include/alsa/input.h" 3
typedef struct _snd_input snd_input_t;


typedef enum _snd_input_type {

 SND_INPUT_STDIO,

 SND_INPUT_BUFFER
} snd_input_type_t;

int snd_input_stdio_open(snd_input_t **inputp, const char *file, const char *mode);
int snd_input_stdio_attach(snd_input_t **inputp, FILE *fp, int _close);
int snd_input_buffer_open(snd_input_t **inputp, const char *buffer, ssize_t size);
int snd_input_close(snd_input_t *input);
int snd_input_scanf(snd_input_t *input, const char *format, ...)

 __attribute__ ((format (scanf, 2, 3)))

 ;
char *snd_input_gets(snd_input_t *input, char *str, size_t size);
int snd_input_getc(snd_input_t *input);
int snd_input_ungetc(snd_input_t *input, int c);
# 46 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/output.h" 1 3
# 54 "/usr/include/alsa/output.h" 3
typedef struct _snd_output snd_output_t;


typedef enum _snd_output_type {

 SND_OUTPUT_STDIO,

 SND_OUTPUT_BUFFER
} snd_output_type_t;

int snd_output_stdio_open(snd_output_t **outputp, const char *file, const char *mode);
int snd_output_stdio_attach(snd_output_t **outputp, FILE *fp, int _close);
int snd_output_buffer_open(snd_output_t **outputp);
size_t snd_output_buffer_string(snd_output_t *output, char **buf);
int snd_output_close(snd_output_t *output);
int snd_output_printf(snd_output_t *output, const char *format, ...)

 __attribute__ ((format (printf, 2, 3)))

 ;
int snd_output_vprintf(snd_output_t *output, const char *format, va_list args);
int snd_output_puts(snd_output_t *output, const char *str);
int snd_output_putc(snd_output_t *output, int c);
int snd_output_flush(snd_output_t *output);
# 47 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/error.h" 1 3
# 45 "/usr/include/alsa/error.h" 3
const char *snd_strerror(int errnum);
# 59 "/usr/include/alsa/error.h" 3
typedef void (*snd_lib_error_handler_t)(const char *file, int line, const char *function, int err, const char *fmt, ...) ;
extern snd_lib_error_handler_t snd_lib_error;
extern int snd_lib_error_set_handler(snd_lib_error_handler_t handler);
# 48 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/conf.h" 1 3
# 48 "/usr/include/alsa/conf.h" 3
typedef enum _snd_config_type {

        SND_CONFIG_TYPE_INTEGER,

        SND_CONFIG_TYPE_INTEGER64,

        SND_CONFIG_TYPE_REAL,

        SND_CONFIG_TYPE_STRING,

        SND_CONFIG_TYPE_POINTER,

 SND_CONFIG_TYPE_COMPOUND = 1024
} snd_config_type_t;







typedef struct _snd_config snd_config_t;







typedef struct _snd_config_iterator *snd_config_iterator_t;





typedef struct _snd_config_update snd_config_update_t;

extern snd_config_t *snd_config;

int snd_config_top(snd_config_t **config);

int snd_config_load(snd_config_t *config, snd_input_t *in);
int snd_config_load_override(snd_config_t *config, snd_input_t *in);
int snd_config_save(snd_config_t *config, snd_output_t *out);
int snd_config_update(void);
int snd_config_update_r(snd_config_t **top, snd_config_update_t **update, const char *path);
int snd_config_update_free(snd_config_update_t *update);
int snd_config_update_free_global(void);

int snd_config_search(snd_config_t *config, const char *key,
        snd_config_t **result);
int snd_config_searchv(snd_config_t *config,
         snd_config_t **result, ...);
int snd_config_search_definition(snd_config_t *config,
     const char *base, const char *key,
     snd_config_t **result);

int snd_config_expand(snd_config_t *config, snd_config_t *root,
        const char *args, snd_config_t *private_data,
        snd_config_t **result);
int snd_config_evaluate(snd_config_t *config, snd_config_t *root,
   snd_config_t *private_data, snd_config_t **result);

int snd_config_add(snd_config_t *config, snd_config_t *leaf);
int snd_config_delete(snd_config_t *config);
int snd_config_delete_compound_members(const snd_config_t *config);
int snd_config_copy(snd_config_t **dst, snd_config_t *src);

int snd_config_make(snd_config_t **config, const char *key,
      snd_config_type_t type);
int snd_config_make_integer(snd_config_t **config, const char *key);
int snd_config_make_integer64(snd_config_t **config, const char *key);
int snd_config_make_real(snd_config_t **config, const char *key);
int snd_config_make_string(snd_config_t **config, const char *key);
int snd_config_make_pointer(snd_config_t **config, const char *key);
int snd_config_make_compound(snd_config_t **config, const char *key, int join);

int snd_config_imake_integer(snd_config_t **config, const char *key, const long value);
int snd_config_imake_integer64(snd_config_t **config, const char *key, const long long value);
int snd_config_imake_real(snd_config_t **config, const char *key, const double value);
int snd_config_imake_string(snd_config_t **config, const char *key, const char *ascii);
int snd_config_imake_pointer(snd_config_t **config, const char *key, const void *ptr);

snd_config_type_t snd_config_get_type(const snd_config_t *config);

int snd_config_set_id(snd_config_t *config, const char *id);
int snd_config_set_integer(snd_config_t *config, long value);
int snd_config_set_integer64(snd_config_t *config, long long value);
int snd_config_set_real(snd_config_t *config, double value);
int snd_config_set_string(snd_config_t *config, const char *value);
int snd_config_set_ascii(snd_config_t *config, const char *ascii);
int snd_config_set_pointer(snd_config_t *config, const void *ptr);
int snd_config_get_id(const snd_config_t *config, const char **value);
int snd_config_get_integer(const snd_config_t *config, long *value);
int snd_config_get_integer64(const snd_config_t *config, long long *value);
int snd_config_get_real(const snd_config_t *config, double *value);
int snd_config_get_ireal(const snd_config_t *config, double *value);
int snd_config_get_string(const snd_config_t *config, const char **value);
int snd_config_get_ascii(const snd_config_t *config, char **value);
int snd_config_get_pointer(const snd_config_t *config, const void **value);
int snd_config_test_id(const snd_config_t *config, const char *id);

snd_config_iterator_t snd_config_iterator_first(const snd_config_t *node);
snd_config_iterator_t snd_config_iterator_next(const snd_config_iterator_t iterator);
snd_config_iterator_t snd_config_iterator_end(const snd_config_t *node);
snd_config_t *snd_config_iterator_entry(const snd_config_iterator_t iterator);
# 168 "/usr/include/alsa/conf.h" 3
int snd_config_get_bool_ascii(const char *ascii);
int snd_config_get_bool(const snd_config_t *conf);
int snd_config_get_ctl_iface_ascii(const char *ascii);
int snd_config_get_ctl_iface(const snd_config_t *conf);






typedef struct snd_devname snd_devname_t;




struct snd_devname {
 char *name;
 char *comment;
 snd_devname_t *next;
};

int snd_names_list(const char *iface, snd_devname_t **list);
void snd_names_list_free(snd_devname_t *list);
# 49 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/pcm.h" 1 3
# 46 "/usr/include/alsa/pcm.h" 3
typedef struct _snd_pcm_info snd_pcm_info_t;

typedef struct _snd_pcm_hw_params snd_pcm_hw_params_t;

typedef struct _snd_pcm_sw_params snd_pcm_sw_params_t;

 typedef struct _snd_pcm_status snd_pcm_status_t;

typedef struct _snd_pcm_access_mask snd_pcm_access_mask_t;

typedef struct _snd_pcm_format_mask snd_pcm_format_mask_t;

typedef struct _snd_pcm_subformat_mask snd_pcm_subformat_mask_t;


typedef enum _snd_pcm_class {


 SND_PCM_CLASS_GENERIC = 0,

 SND_PCM_CLASS_MULTI,

 SND_PCM_CLASS_MODEM,

 SND_PCM_CLASS_DIGITIZER,
 SND_PCM_CLASS_LAST = SND_PCM_CLASS_DIGITIZER
} snd_pcm_class_t;


typedef enum _snd_pcm_subclass {

 SND_PCM_SUBCLASS_GENERIC_MIX = 0,

 SND_PCM_SUBCLASS_MULTI_MIX,
 SND_PCM_SUBCLASS_LAST = SND_PCM_SUBCLASS_MULTI_MIX
} snd_pcm_subclass_t;


typedef enum _snd_pcm_stream {

 SND_PCM_STREAM_PLAYBACK = 0,

 SND_PCM_STREAM_CAPTURE,
 SND_PCM_STREAM_LAST = SND_PCM_STREAM_CAPTURE
} snd_pcm_stream_t;


typedef enum _snd_pcm_access {

 SND_PCM_ACCESS_MMAP_INTERLEAVED = 0,

 SND_PCM_ACCESS_MMAP_NONINTERLEAVED,

 SND_PCM_ACCESS_MMAP_COMPLEX,

 SND_PCM_ACCESS_RW_INTERLEAVED,

 SND_PCM_ACCESS_RW_NONINTERLEAVED,
 SND_PCM_ACCESS_LAST = SND_PCM_ACCESS_RW_NONINTERLEAVED
} snd_pcm_access_t;


typedef enum _snd_pcm_format {

 SND_PCM_FORMAT_UNKNOWN = -1,

 SND_PCM_FORMAT_S8 = 0,

 SND_PCM_FORMAT_U8,

 SND_PCM_FORMAT_S16_LE,

 SND_PCM_FORMAT_S16_BE,

 SND_PCM_FORMAT_U16_LE,

 SND_PCM_FORMAT_U16_BE,

 SND_PCM_FORMAT_S24_LE,

 SND_PCM_FORMAT_S24_BE,

 SND_PCM_FORMAT_U24_LE,

 SND_PCM_FORMAT_U24_BE,

 SND_PCM_FORMAT_S32_LE,

 SND_PCM_FORMAT_S32_BE,

 SND_PCM_FORMAT_U32_LE,

 SND_PCM_FORMAT_U32_BE,

 SND_PCM_FORMAT_FLOAT_LE,

 SND_PCM_FORMAT_FLOAT_BE,

 SND_PCM_FORMAT_FLOAT64_LE,

 SND_PCM_FORMAT_FLOAT64_BE,

 SND_PCM_FORMAT_IEC958_SUBFRAME_LE,

 SND_PCM_FORMAT_IEC958_SUBFRAME_BE,

 SND_PCM_FORMAT_MU_LAW,

 SND_PCM_FORMAT_A_LAW,

 SND_PCM_FORMAT_IMA_ADPCM,

 SND_PCM_FORMAT_MPEG,

 SND_PCM_FORMAT_GSM,

 SND_PCM_FORMAT_SPECIAL = 31,

 SND_PCM_FORMAT_S24_3LE = 32,

 SND_PCM_FORMAT_S24_3BE,

 SND_PCM_FORMAT_U24_3LE,

 SND_PCM_FORMAT_U24_3BE,

 SND_PCM_FORMAT_S20_3LE,

 SND_PCM_FORMAT_S20_3BE,

 SND_PCM_FORMAT_U20_3LE,

 SND_PCM_FORMAT_U20_3BE,

 SND_PCM_FORMAT_S18_3LE,

 SND_PCM_FORMAT_S18_3BE,

 SND_PCM_FORMAT_U18_3LE,

 SND_PCM_FORMAT_U18_3BE,
 SND_PCM_FORMAT_LAST = SND_PCM_FORMAT_U18_3BE,



 SND_PCM_FORMAT_S16 = SND_PCM_FORMAT_S16_LE,

 SND_PCM_FORMAT_U16 = SND_PCM_FORMAT_U16_LE,

 SND_PCM_FORMAT_S24 = SND_PCM_FORMAT_S24_LE,

 SND_PCM_FORMAT_U24 = SND_PCM_FORMAT_U24_LE,

 SND_PCM_FORMAT_S32 = SND_PCM_FORMAT_S32_LE,

 SND_PCM_FORMAT_U32 = SND_PCM_FORMAT_U32_LE,

 SND_PCM_FORMAT_FLOAT = SND_PCM_FORMAT_FLOAT_LE,

 SND_PCM_FORMAT_FLOAT64 = SND_PCM_FORMAT_FLOAT64_LE,

 SND_PCM_FORMAT_IEC958_SUBFRAME = SND_PCM_FORMAT_IEC958_SUBFRAME_LE
# 230 "/usr/include/alsa/pcm.h" 3
} snd_pcm_format_t;


typedef enum _snd_pcm_subformat {

 SND_PCM_SUBFORMAT_STD = 0,
 SND_PCM_SUBFORMAT_LAST = SND_PCM_SUBFORMAT_STD
} snd_pcm_subformat_t;


typedef enum _snd_pcm_state {

 SND_PCM_STATE_OPEN = 0,

 SND_PCM_STATE_SETUP,

 SND_PCM_STATE_PREPARED,

 SND_PCM_STATE_RUNNING,

 SND_PCM_STATE_XRUN,

 SND_PCM_STATE_DRAINING,

 SND_PCM_STATE_PAUSED,

 SND_PCM_STATE_SUSPENDED,

 SND_PCM_STATE_DISCONNECTED,
 SND_PCM_STATE_LAST = SND_PCM_STATE_DISCONNECTED
} snd_pcm_state_t;


typedef enum _snd_pcm_start {

 SND_PCM_START_DATA = 0,

 SND_PCM_START_EXPLICIT,
 SND_PCM_START_LAST = SND_PCM_START_EXPLICIT
} snd_pcm_start_t;


typedef enum _snd_pcm_xrun {

 SND_PCM_XRUN_NONE = 0,

 SND_PCM_XRUN_STOP,
 SND_PCM_XRUN_LAST = SND_PCM_XRUN_STOP
} snd_pcm_xrun_t;


typedef enum _snd_pcm_tstamp {

 SND_PCM_TSTAMP_NONE = 0,

 SND_PCM_TSTAMP_ENABLE,



 SND_PCM_TSTAMP_MMAP = SND_PCM_TSTAMP_ENABLE,
 SND_PCM_TSTAMP_LAST = SND_PCM_TSTAMP_ENABLE
} snd_pcm_tstamp_t;


typedef unsigned long snd_pcm_uframes_t;

typedef long snd_pcm_sframes_t;
# 312 "/usr/include/alsa/pcm.h" 3
typedef struct _snd_pcm snd_pcm_t;


enum _snd_pcm_type {

 SND_PCM_TYPE_HW = 0,

 SND_PCM_TYPE_HOOKS,


 SND_PCM_TYPE_MULTI,

 SND_PCM_TYPE_FILE,

 SND_PCM_TYPE_NULL,

 SND_PCM_TYPE_SHM,

 SND_PCM_TYPE_INET,

 SND_PCM_TYPE_COPY,

 SND_PCM_TYPE_LINEAR,

 SND_PCM_TYPE_ALAW,

 SND_PCM_TYPE_MULAW,

 SND_PCM_TYPE_ADPCM,

 SND_PCM_TYPE_RATE,

 SND_PCM_TYPE_ROUTE,

 SND_PCM_TYPE_PLUG,

 SND_PCM_TYPE_SHARE,

 SND_PCM_TYPE_METER,

 SND_PCM_TYPE_MIX,

 SND_PCM_TYPE_DROUTE,

 SND_PCM_TYPE_LBSERVER,

 SND_PCM_TYPE_LINEAR_FLOAT,

 SND_PCM_TYPE_LADSPA,

 SND_PCM_TYPE_DMIX,

 SND_PCM_TYPE_JACK,

 SND_PCM_TYPE_DSNOOP,

 SND_PCM_TYPE_DSHARE,

 SND_PCM_TYPE_IEC958,

 SND_PCM_TYPE_SOFTVOL,

 SND_PCM_TYPE_IOPLUG,

 SND_PCM_TYPE_EXTPLUG,

 SND_PCM_TYPE_MMAP_EMUL,
 SND_PCM_TYPE_LAST = SND_PCM_TYPE_MMAP_EMUL
};


typedef enum _snd_pcm_type snd_pcm_type_t;


typedef struct _snd_pcm_channel_area {

 void *addr;

 unsigned int first;

 unsigned int step;
} snd_pcm_channel_area_t;


typedef union _snd_pcm_sync_id {

 unsigned char id[16];

 unsigned short id16[8];

 unsigned int id32[4];
} snd_pcm_sync_id_t;


typedef struct _snd_pcm_scope snd_pcm_scope_t;

int snd_pcm_open(snd_pcm_t **pcm, const char *name,
   snd_pcm_stream_t stream, int mode);
int snd_pcm_open_lconf(snd_pcm_t **pcm, const char *name,
         snd_pcm_stream_t stream, int mode,
         snd_config_t *lconf);

int snd_pcm_close(snd_pcm_t *pcm);
const char *snd_pcm_name(snd_pcm_t *pcm);
snd_pcm_type_t snd_pcm_type(snd_pcm_t *pcm);
snd_pcm_stream_t snd_pcm_stream(snd_pcm_t *pcm);
int snd_pcm_poll_descriptors_count(snd_pcm_t *pcm);
int snd_pcm_poll_descriptors(snd_pcm_t *pcm, struct pollfd *pfds, unsigned int space);
int snd_pcm_poll_descriptors_revents(snd_pcm_t *pcm, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
int snd_pcm_nonblock(snd_pcm_t *pcm, int nonblock);
int snd_async_add_pcm_handler(snd_async_handler_t **handler, snd_pcm_t *pcm,
         snd_async_callback_t callback, void *private_data);
snd_pcm_t *snd_async_handler_get_pcm(snd_async_handler_t *handler);
int snd_pcm_info(snd_pcm_t *pcm, snd_pcm_info_t *info);
int snd_pcm_hw_params_current(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);
int snd_pcm_hw_params(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);
int snd_pcm_hw_free(snd_pcm_t *pcm);
int snd_pcm_sw_params_current(snd_pcm_t *pcm, snd_pcm_sw_params_t *params);
int snd_pcm_sw_params(snd_pcm_t *pcm, snd_pcm_sw_params_t *params);
int snd_pcm_prepare(snd_pcm_t *pcm);
int snd_pcm_reset(snd_pcm_t *pcm);
int snd_pcm_status(snd_pcm_t *pcm, snd_pcm_status_t *status);
int snd_pcm_start(snd_pcm_t *pcm);
int snd_pcm_drop(snd_pcm_t *pcm);
int snd_pcm_drain(snd_pcm_t *pcm);
int snd_pcm_pause(snd_pcm_t *pcm, int enable);
snd_pcm_state_t snd_pcm_state(snd_pcm_t *pcm);
int snd_pcm_hwsync(snd_pcm_t *pcm);
int snd_pcm_delay(snd_pcm_t *pcm, snd_pcm_sframes_t *delayp);
int snd_pcm_resume(snd_pcm_t *pcm);
int snd_pcm_htimestamp(snd_pcm_t *pcm, snd_pcm_uframes_t *avail, snd_htimestamp_t *tstamp);
snd_pcm_sframes_t snd_pcm_avail_update(snd_pcm_t *pcm);
snd_pcm_sframes_t snd_pcm_rewindable(snd_pcm_t *pcm);
snd_pcm_sframes_t snd_pcm_rewind(snd_pcm_t *pcm, snd_pcm_uframes_t frames);
snd_pcm_sframes_t snd_pcm_forwardable(snd_pcm_t *pcm);
snd_pcm_sframes_t snd_pcm_forward(snd_pcm_t *pcm, snd_pcm_uframes_t frames);
snd_pcm_sframes_t snd_pcm_writei(snd_pcm_t *pcm, const void *buffer, snd_pcm_uframes_t size);
snd_pcm_sframes_t snd_pcm_readi(snd_pcm_t *pcm, void *buffer, snd_pcm_uframes_t size);
snd_pcm_sframes_t snd_pcm_writen(snd_pcm_t *pcm, void **bufs, snd_pcm_uframes_t size);
snd_pcm_sframes_t snd_pcm_readn(snd_pcm_t *pcm, void **bufs, snd_pcm_uframes_t size);
int snd_pcm_wait(snd_pcm_t *pcm, int timeout);

int snd_pcm_link(snd_pcm_t *pcm1, snd_pcm_t *pcm2);
int snd_pcm_unlink(snd_pcm_t *pcm);






int snd_pcm_recover(snd_pcm_t *pcm, int err, int silent);
int snd_pcm_set_params(snd_pcm_t *pcm,
                       snd_pcm_format_t format,
                       snd_pcm_access_t access,
                       unsigned int channels,
                       unsigned int rate,
                       int soft_resample,
                       unsigned int latency);
int snd_pcm_get_params(snd_pcm_t *pcm,
                       snd_pcm_uframes_t *buffer_size,
                       snd_pcm_uframes_t *period_size);
# 483 "/usr/include/alsa/pcm.h" 3
size_t snd_pcm_info_sizeof(void);





int snd_pcm_info_malloc(snd_pcm_info_t **ptr);
void snd_pcm_info_free(snd_pcm_info_t *obj);
void snd_pcm_info_copy(snd_pcm_info_t *dst, const snd_pcm_info_t *src);
unsigned int snd_pcm_info_get_device(const snd_pcm_info_t *obj);
unsigned int snd_pcm_info_get_subdevice(const snd_pcm_info_t *obj);
snd_pcm_stream_t snd_pcm_info_get_stream(const snd_pcm_info_t *obj);
int snd_pcm_info_get_card(const snd_pcm_info_t *obj);
const char *snd_pcm_info_get_id(const snd_pcm_info_t *obj);
const char *snd_pcm_info_get_name(const snd_pcm_info_t *obj);
const char *snd_pcm_info_get_subdevice_name(const snd_pcm_info_t *obj);
snd_pcm_class_t snd_pcm_info_get_class(const snd_pcm_info_t *obj);
snd_pcm_subclass_t snd_pcm_info_get_subclass(const snd_pcm_info_t *obj);
unsigned int snd_pcm_info_get_subdevices_count(const snd_pcm_info_t *obj);
unsigned int snd_pcm_info_get_subdevices_avail(const snd_pcm_info_t *obj);
snd_pcm_sync_id_t snd_pcm_info_get_sync(const snd_pcm_info_t *obj);
void snd_pcm_info_set_device(snd_pcm_info_t *obj, unsigned int val);
void snd_pcm_info_set_subdevice(snd_pcm_info_t *obj, unsigned int val);
void snd_pcm_info_set_stream(snd_pcm_info_t *obj, snd_pcm_stream_t val);
# 517 "/usr/include/alsa/pcm.h" 3
int snd_pcm_hw_params_any(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);

int snd_pcm_hw_params_can_mmap_sample_resolution(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_is_double(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_is_batch(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_is_block_transfer(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_is_monotonic(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_can_overrange(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_can_forward(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_can_rewind(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_can_pause(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_can_resume(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_is_half_duplex(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_is_joint_duplex(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_can_sync_start(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_get_rate_numden(const snd_pcm_hw_params_t *params,
          unsigned int *rate_num,
          unsigned int *rate_den);
int snd_pcm_hw_params_get_sbits(const snd_pcm_hw_params_t *params);
int snd_pcm_hw_params_get_fifo_size(const snd_pcm_hw_params_t *params);
# 564 "/usr/include/alsa/pcm.h" 3
size_t snd_pcm_hw_params_sizeof(void);





int snd_pcm_hw_params_malloc(snd_pcm_hw_params_t **ptr);
void snd_pcm_hw_params_free(snd_pcm_hw_params_t *obj);
void snd_pcm_hw_params_copy(snd_pcm_hw_params_t *dst, const snd_pcm_hw_params_t *src);



int snd_pcm_hw_params_get_access(const snd_pcm_hw_params_t *params, snd_pcm_access_t *_access);
int snd_pcm_hw_params_test_access(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_access_t _access);
int snd_pcm_hw_params_set_access(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_access_t _access);
int snd_pcm_hw_params_set_access_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_access_t *_access);
int snd_pcm_hw_params_set_access_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_access_t *_access);
int snd_pcm_hw_params_set_access_mask(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_access_mask_t *mask);
int snd_pcm_hw_params_get_access_mask(snd_pcm_hw_params_t *params, snd_pcm_access_mask_t *mask);

int snd_pcm_hw_params_get_format(const snd_pcm_hw_params_t *params, snd_pcm_format_t *val);
int snd_pcm_hw_params_test_format(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_format_t val);
int snd_pcm_hw_params_set_format(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_format_t val);
int snd_pcm_hw_params_set_format_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_format_t *format);
int snd_pcm_hw_params_set_format_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_format_t *format);
int snd_pcm_hw_params_set_format_mask(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_format_mask_t *mask);
void snd_pcm_hw_params_get_format_mask(snd_pcm_hw_params_t *params, snd_pcm_format_mask_t *mask);

int snd_pcm_hw_params_get_subformat(const snd_pcm_hw_params_t *params, snd_pcm_subformat_t *subformat);
int snd_pcm_hw_params_test_subformat(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_subformat_t subformat);
int snd_pcm_hw_params_set_subformat(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_subformat_t subformat);
int snd_pcm_hw_params_set_subformat_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_subformat_t *subformat);
int snd_pcm_hw_params_set_subformat_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_subformat_t *subformat);
int snd_pcm_hw_params_set_subformat_mask(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_subformat_mask_t *mask);
void snd_pcm_hw_params_get_subformat_mask(snd_pcm_hw_params_t *params, snd_pcm_subformat_mask_t *mask);

int snd_pcm_hw_params_get_channels(const snd_pcm_hw_params_t *params, unsigned int *val);
int snd_pcm_hw_params_get_channels_min(const snd_pcm_hw_params_t *params, unsigned int *val);
int snd_pcm_hw_params_get_channels_max(const snd_pcm_hw_params_t *params, unsigned int *val);
int snd_pcm_hw_params_test_channels(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val);
int snd_pcm_hw_params_set_channels(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val);
int snd_pcm_hw_params_set_channels_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val);
int snd_pcm_hw_params_set_channels_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val);
int snd_pcm_hw_params_set_channels_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, unsigned int *max);
int snd_pcm_hw_params_set_channels_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val);
int snd_pcm_hw_params_set_channels_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val);
int snd_pcm_hw_params_set_channels_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val);

int snd_pcm_hw_params_get_rate(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_get_rate_min(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_get_rate_max(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_test_rate(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_rate(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_rate_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_rate_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_rate_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, int *mindir, unsigned int *max, int *maxdir);
int snd_pcm_hw_params_set_rate_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_rate_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_rate_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_rate_resample(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val);
int snd_pcm_hw_params_get_rate_resample(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val);
int snd_pcm_hw_params_set_export_buffer(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val);
int snd_pcm_hw_params_get_export_buffer(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val);

int snd_pcm_hw_params_get_period_time(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_get_period_time_min(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_get_period_time_max(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_test_period_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_period_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_period_time_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_period_time_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_period_time_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, int *mindir, unsigned int *max, int *maxdir);
int snd_pcm_hw_params_set_period_time_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_period_time_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_period_time_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);

int snd_pcm_hw_params_get_period_size(const snd_pcm_hw_params_t *params, snd_pcm_uframes_t *frames, int *dir);
int snd_pcm_hw_params_get_period_size_min(const snd_pcm_hw_params_t *params, snd_pcm_uframes_t *frames, int *dir);
int snd_pcm_hw_params_get_period_size_max(const snd_pcm_hw_params_t *params, snd_pcm_uframes_t *frames, int *dir);
int snd_pcm_hw_params_test_period_size(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t val, int dir);
int snd_pcm_hw_params_set_period_size(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t val, int dir);
int snd_pcm_hw_params_set_period_size_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val, int *dir);
int snd_pcm_hw_params_set_period_size_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val, int *dir);
int snd_pcm_hw_params_set_period_size_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *min, int *mindir, snd_pcm_uframes_t *max, int *maxdir);
int snd_pcm_hw_params_set_period_size_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val, int *dir);
int snd_pcm_hw_params_set_period_size_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val, int *dir);
int snd_pcm_hw_params_set_period_size_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val, int *dir);
int snd_pcm_hw_params_set_period_size_integer(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);

int snd_pcm_hw_params_get_periods(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_get_periods_min(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_get_periods_max(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_test_periods(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_periods(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_periods_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_periods_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_periods_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, int *mindir, unsigned int *max, int *maxdir);
int snd_pcm_hw_params_set_periods_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_periods_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_periods_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_periods_integer(snd_pcm_t *pcm, snd_pcm_hw_params_t *params);

int snd_pcm_hw_params_get_buffer_time(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_get_buffer_time_min(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_get_buffer_time_max(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_test_buffer_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_buffer_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir);
int snd_pcm_hw_params_set_buffer_time_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_buffer_time_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_buffer_time_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, int *mindir, unsigned int *max, int *maxdir);
int snd_pcm_hw_params_set_buffer_time_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_buffer_time_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);
int snd_pcm_hw_params_set_buffer_time_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir);

int snd_pcm_hw_params_get_buffer_size(const snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_hw_params_get_buffer_size_min(const snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_hw_params_get_buffer_size_max(const snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_hw_params_test_buffer_size(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t val);
int snd_pcm_hw_params_set_buffer_size(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t val);
int snd_pcm_hw_params_set_buffer_size_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_hw_params_set_buffer_size_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_hw_params_set_buffer_size_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *min, snd_pcm_uframes_t *max);
int snd_pcm_hw_params_set_buffer_size_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_hw_params_set_buffer_size_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_hw_params_set_buffer_size_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val);



int snd_pcm_hw_params_get_min_align(const snd_pcm_hw_params_t *params, snd_pcm_uframes_t *val);
# 703 "/usr/include/alsa/pcm.h" 3
size_t snd_pcm_sw_params_sizeof(void);





int snd_pcm_sw_params_malloc(snd_pcm_sw_params_t **ptr);
void snd_pcm_sw_params_free(snd_pcm_sw_params_t *obj);
void snd_pcm_sw_params_copy(snd_pcm_sw_params_t *dst, const snd_pcm_sw_params_t *src);
int snd_pcm_sw_params_get_boundary(const snd_pcm_sw_params_t *params, snd_pcm_uframes_t *val);



int snd_pcm_sw_params_set_tstamp_mode(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_tstamp_t val);
int snd_pcm_sw_params_get_tstamp_mode(const snd_pcm_sw_params_t *params, snd_pcm_tstamp_t *val);
int snd_pcm_sw_params_set_avail_min(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val);
int snd_pcm_sw_params_get_avail_min(const snd_pcm_sw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_sw_params_set_period_event(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, int val);
int snd_pcm_sw_params_get_period_event(const snd_pcm_sw_params_t *params, int *val);
int snd_pcm_sw_params_set_start_threshold(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val);
int snd_pcm_sw_params_get_start_threshold(const snd_pcm_sw_params_t *paramsm, snd_pcm_uframes_t *val);
int snd_pcm_sw_params_set_stop_threshold(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val);
int snd_pcm_sw_params_get_stop_threshold(const snd_pcm_sw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_sw_params_set_silence_threshold(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val);
int snd_pcm_sw_params_get_silence_threshold(const snd_pcm_sw_params_t *params, snd_pcm_uframes_t *val);
int snd_pcm_sw_params_set_silence_size(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val);
int snd_pcm_sw_params_get_silence_size(const snd_pcm_sw_params_t *params, snd_pcm_uframes_t *val);
# 749 "/usr/include/alsa/pcm.h" 3
size_t snd_pcm_access_mask_sizeof(void);





int snd_pcm_access_mask_malloc(snd_pcm_access_mask_t **ptr);
void snd_pcm_access_mask_free(snd_pcm_access_mask_t *obj);
void snd_pcm_access_mask_copy(snd_pcm_access_mask_t *dst, const snd_pcm_access_mask_t *src);
void snd_pcm_access_mask_none(snd_pcm_access_mask_t *mask);
void snd_pcm_access_mask_any(snd_pcm_access_mask_t *mask);
int snd_pcm_access_mask_test(const snd_pcm_access_mask_t *mask, snd_pcm_access_t val);
int snd_pcm_access_mask_empty(const snd_pcm_access_mask_t *mask);
void snd_pcm_access_mask_set(snd_pcm_access_mask_t *mask, snd_pcm_access_t val);
void snd_pcm_access_mask_reset(snd_pcm_access_mask_t *mask, snd_pcm_access_t val);
# 774 "/usr/include/alsa/pcm.h" 3
size_t snd_pcm_format_mask_sizeof(void);





int snd_pcm_format_mask_malloc(snd_pcm_format_mask_t **ptr);
void snd_pcm_format_mask_free(snd_pcm_format_mask_t *obj);
void snd_pcm_format_mask_copy(snd_pcm_format_mask_t *dst, const snd_pcm_format_mask_t *src);
void snd_pcm_format_mask_none(snd_pcm_format_mask_t *mask);
void snd_pcm_format_mask_any(snd_pcm_format_mask_t *mask);
int snd_pcm_format_mask_test(const snd_pcm_format_mask_t *mask, snd_pcm_format_t val);
int snd_pcm_format_mask_empty(const snd_pcm_format_mask_t *mask);
void snd_pcm_format_mask_set(snd_pcm_format_mask_t *mask, snd_pcm_format_t val);
void snd_pcm_format_mask_reset(snd_pcm_format_mask_t *mask, snd_pcm_format_t val);
# 799 "/usr/include/alsa/pcm.h" 3
size_t snd_pcm_subformat_mask_sizeof(void);





int snd_pcm_subformat_mask_malloc(snd_pcm_subformat_mask_t **ptr);
void snd_pcm_subformat_mask_free(snd_pcm_subformat_mask_t *obj);
void snd_pcm_subformat_mask_copy(snd_pcm_subformat_mask_t *dst, const snd_pcm_subformat_mask_t *src);
void snd_pcm_subformat_mask_none(snd_pcm_subformat_mask_t *mask);
void snd_pcm_subformat_mask_any(snd_pcm_subformat_mask_t *mask);
int snd_pcm_subformat_mask_test(const snd_pcm_subformat_mask_t *mask, snd_pcm_subformat_t val);
int snd_pcm_subformat_mask_empty(const snd_pcm_subformat_mask_t *mask);
void snd_pcm_subformat_mask_set(snd_pcm_subformat_mask_t *mask, snd_pcm_subformat_t val);
void snd_pcm_subformat_mask_reset(snd_pcm_subformat_mask_t *mask, snd_pcm_subformat_t val);
# 824 "/usr/include/alsa/pcm.h" 3
size_t snd_pcm_status_sizeof(void);





int snd_pcm_status_malloc(snd_pcm_status_t **ptr);
void snd_pcm_status_free(snd_pcm_status_t *obj);
void snd_pcm_status_copy(snd_pcm_status_t *dst, const snd_pcm_status_t *src);
snd_pcm_state_t snd_pcm_status_get_state(const snd_pcm_status_t *obj);
void snd_pcm_status_get_trigger_tstamp(const snd_pcm_status_t *obj, snd_timestamp_t *ptr);
void snd_pcm_status_get_trigger_htstamp(const snd_pcm_status_t *obj, snd_htimestamp_t *ptr);
void snd_pcm_status_get_tstamp(const snd_pcm_status_t *obj, snd_timestamp_t *ptr);
void snd_pcm_status_get_htstamp(const snd_pcm_status_t *obj, snd_htimestamp_t *ptr);
snd_pcm_sframes_t snd_pcm_status_get_delay(const snd_pcm_status_t *obj);
snd_pcm_uframes_t snd_pcm_status_get_avail(const snd_pcm_status_t *obj);
snd_pcm_uframes_t snd_pcm_status_get_avail_max(const snd_pcm_status_t *obj);
snd_pcm_uframes_t snd_pcm_status_get_overrange(const snd_pcm_status_t *obj);
# 852 "/usr/include/alsa/pcm.h" 3
const char *snd_pcm_type_name(snd_pcm_type_t type);
const char *snd_pcm_stream_name(const snd_pcm_stream_t stream);
const char *snd_pcm_access_name(const snd_pcm_access_t _access);
const char *snd_pcm_format_name(const snd_pcm_format_t format);
const char *snd_pcm_format_description(const snd_pcm_format_t format);
const char *snd_pcm_subformat_name(const snd_pcm_subformat_t subformat);
const char *snd_pcm_subformat_description(const snd_pcm_subformat_t subformat);
snd_pcm_format_t snd_pcm_format_value(const char* name);
const char *snd_pcm_tstamp_mode_name(const snd_pcm_tstamp_t mode);
const char *snd_pcm_state_name(const snd_pcm_state_t state);
# 872 "/usr/include/alsa/pcm.h" 3
int snd_pcm_dump(snd_pcm_t *pcm, snd_output_t *out);
int snd_pcm_dump_hw_setup(snd_pcm_t *pcm, snd_output_t *out);
int snd_pcm_dump_sw_setup(snd_pcm_t *pcm, snd_output_t *out);
int snd_pcm_dump_setup(snd_pcm_t *pcm, snd_output_t *out);
int snd_pcm_hw_params_dump(snd_pcm_hw_params_t *params, snd_output_t *out);
int snd_pcm_sw_params_dump(snd_pcm_sw_params_t *params, snd_output_t *out);
int snd_pcm_status_dump(snd_pcm_status_t *status, snd_output_t *out);
# 889 "/usr/include/alsa/pcm.h" 3
int snd_pcm_mmap_begin(snd_pcm_t *pcm,
         const snd_pcm_channel_area_t **areas,
         snd_pcm_uframes_t *offset,
         snd_pcm_uframes_t *frames);
snd_pcm_sframes_t snd_pcm_mmap_commit(snd_pcm_t *pcm,
          snd_pcm_uframes_t offset,
          snd_pcm_uframes_t frames);
snd_pcm_sframes_t snd_pcm_mmap_writei(snd_pcm_t *pcm, const void *buffer, snd_pcm_uframes_t size);
snd_pcm_sframes_t snd_pcm_mmap_readi(snd_pcm_t *pcm, void *buffer, snd_pcm_uframes_t size);
snd_pcm_sframes_t snd_pcm_mmap_writen(snd_pcm_t *pcm, void **bufs, snd_pcm_uframes_t size);
snd_pcm_sframes_t snd_pcm_mmap_readn(snd_pcm_t *pcm, void **bufs, snd_pcm_uframes_t size);
# 910 "/usr/include/alsa/pcm.h" 3
int snd_pcm_format_signed(snd_pcm_format_t format);
int snd_pcm_format_unsigned(snd_pcm_format_t format);
int snd_pcm_format_linear(snd_pcm_format_t format);
int snd_pcm_format_float(snd_pcm_format_t format);
int snd_pcm_format_little_endian(snd_pcm_format_t format);
int snd_pcm_format_big_endian(snd_pcm_format_t format);
int snd_pcm_format_cpu_endian(snd_pcm_format_t format);
int snd_pcm_format_width(snd_pcm_format_t format);
int snd_pcm_format_physical_width(snd_pcm_format_t format);
snd_pcm_format_t snd_pcm_build_linear_format(int width, int pwidth, int unsignd, int big_endian);
ssize_t snd_pcm_format_size(snd_pcm_format_t format, size_t samples);
u_int8_t snd_pcm_format_silence(snd_pcm_format_t format);
u_int16_t snd_pcm_format_silence_16(snd_pcm_format_t format);
u_int32_t snd_pcm_format_silence_32(snd_pcm_format_t format);
u_int64_t snd_pcm_format_silence_64(snd_pcm_format_t format);
int snd_pcm_format_set_silence(snd_pcm_format_t format, void *buf, unsigned int samples);

snd_pcm_sframes_t snd_pcm_bytes_to_frames(snd_pcm_t *pcm, ssize_t bytes);
ssize_t snd_pcm_frames_to_bytes(snd_pcm_t *pcm, snd_pcm_sframes_t frames);
long snd_pcm_bytes_to_samples(snd_pcm_t *pcm, ssize_t bytes);
ssize_t snd_pcm_samples_to_bytes(snd_pcm_t *pcm, long samples);

int snd_pcm_area_silence(const snd_pcm_channel_area_t *dst_channel, snd_pcm_uframes_t dst_offset,
    unsigned int samples, snd_pcm_format_t format);
int snd_pcm_areas_silence(const snd_pcm_channel_area_t *dst_channels, snd_pcm_uframes_t dst_offset,
     unsigned int channels, snd_pcm_uframes_t frames, snd_pcm_format_t format);
int snd_pcm_area_copy(const snd_pcm_channel_area_t *dst_channel, snd_pcm_uframes_t dst_offset,
        const snd_pcm_channel_area_t *src_channel, snd_pcm_uframes_t src_offset,
        unsigned int samples, snd_pcm_format_t format);
int snd_pcm_areas_copy(const snd_pcm_channel_area_t *dst_channels, snd_pcm_uframes_t dst_offset,
         const snd_pcm_channel_area_t *src_channels, snd_pcm_uframes_t src_offset,
         unsigned int channels, snd_pcm_uframes_t frames, snd_pcm_format_t format);
# 953 "/usr/include/alsa/pcm.h" 3
typedef enum _snd_pcm_hook_type {
 SND_PCM_HOOK_TYPE_HW_PARAMS = 0,
 SND_PCM_HOOK_TYPE_HW_FREE,
 SND_PCM_HOOK_TYPE_CLOSE,
 SND_PCM_HOOK_TYPE_LAST = SND_PCM_HOOK_TYPE_CLOSE
} snd_pcm_hook_type_t;


typedef struct _snd_pcm_hook snd_pcm_hook_t;

typedef int (*snd_pcm_hook_func_t)(snd_pcm_hook_t *hook);
snd_pcm_t *snd_pcm_hook_get_pcm(snd_pcm_hook_t *hook);
void *snd_pcm_hook_get_private(snd_pcm_hook_t *hook);
void snd_pcm_hook_set_private(snd_pcm_hook_t *hook, void *private_data);
int snd_pcm_hook_add(snd_pcm_hook_t **hookp, snd_pcm_t *pcm,
       snd_pcm_hook_type_t type,
       snd_pcm_hook_func_t func, void *private_data);
int snd_pcm_hook_remove(snd_pcm_hook_t *hook);
# 982 "/usr/include/alsa/pcm.h" 3
typedef struct _snd_pcm_scope_ops {



 int (*enable)(snd_pcm_scope_t *scope);



 void (*disable)(snd_pcm_scope_t *scope);



 void (*start)(snd_pcm_scope_t *scope);



 void (*stop)(snd_pcm_scope_t *scope);



 void (*update)(snd_pcm_scope_t *scope);



 void (*reset)(snd_pcm_scope_t *scope);



 void (*close)(snd_pcm_scope_t *scope);
} snd_pcm_scope_ops_t;

snd_pcm_uframes_t snd_pcm_meter_get_bufsize(snd_pcm_t *pcm);
unsigned int snd_pcm_meter_get_channels(snd_pcm_t *pcm);
unsigned int snd_pcm_meter_get_rate(snd_pcm_t *pcm);
snd_pcm_uframes_t snd_pcm_meter_get_now(snd_pcm_t *pcm);
snd_pcm_uframes_t snd_pcm_meter_get_boundary(snd_pcm_t *pcm);
int snd_pcm_meter_add_scope(snd_pcm_t *pcm, snd_pcm_scope_t *scope);
snd_pcm_scope_t *snd_pcm_meter_search_scope(snd_pcm_t *pcm, const char *name);
int snd_pcm_scope_malloc(snd_pcm_scope_t **ptr);
void snd_pcm_scope_set_ops(snd_pcm_scope_t *scope, snd_pcm_scope_ops_t *val);
void snd_pcm_scope_set_name(snd_pcm_scope_t *scope, const char *val);
const char *snd_pcm_scope_get_name(snd_pcm_scope_t *scope);
void *snd_pcm_scope_get_callback_private(snd_pcm_scope_t *scope);
void snd_pcm_scope_set_callback_private(snd_pcm_scope_t *scope, void *val);
int snd_pcm_scope_s16_open(snd_pcm_t *pcm, const char *name,
      snd_pcm_scope_t **scopep);
int16_t *snd_pcm_scope_s16_get_channel_buffer(snd_pcm_scope_t *scope,
           unsigned int channel);
# 1041 "/usr/include/alsa/pcm.h" 3
typedef enum _snd_spcm_latency {


 SND_SPCM_LATENCY_STANDARD = 0,


 SND_SPCM_LATENCY_MEDIUM,


 SND_SPCM_LATENCY_REALTIME
} snd_spcm_latency_t;


typedef enum _snd_spcm_xrun_type {

 SND_SPCM_XRUN_IGNORE = 0,

 SND_SPCM_XRUN_STOP
} snd_spcm_xrun_type_t;


typedef enum _snd_spcm_duplex_type {

 SND_SPCM_DUPLEX_LIBERAL = 0,

 SND_SPCM_DUPLEX_PEDANTIC
} snd_spcm_duplex_type_t;

int snd_spcm_init(snd_pcm_t *pcm,
    unsigned int rate,
    unsigned int channels,
    snd_pcm_format_t format,
    snd_pcm_subformat_t subformat,
    snd_spcm_latency_t latency,
    snd_pcm_access_t _access,
    snd_spcm_xrun_type_t xrun_type);

int snd_spcm_init_duplex(snd_pcm_t *playback_pcm,
    snd_pcm_t *capture_pcm,
    unsigned int rate,
    unsigned int channels,
    snd_pcm_format_t format,
    snd_pcm_subformat_t subformat,
    snd_spcm_latency_t latency,
    snd_pcm_access_t _access,
    snd_spcm_xrun_type_t xrun_type,
    snd_spcm_duplex_type_t duplex_type);

int snd_spcm_init_get_params(snd_pcm_t *pcm,
        unsigned int *rate,
        snd_pcm_uframes_t *buffer_size,
        snd_pcm_uframes_t *period_size);
# 1104 "/usr/include/alsa/pcm.h" 3
const char *snd_pcm_start_mode_name(snd_pcm_start_t mode) __attribute__((deprecated));
const char *snd_pcm_xrun_mode_name(snd_pcm_xrun_t mode) __attribute__((deprecated));
int snd_pcm_sw_params_set_start_mode(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_start_t val) __attribute__((deprecated));
snd_pcm_start_t snd_pcm_sw_params_get_start_mode(const snd_pcm_sw_params_t *params) __attribute__((deprecated));
int snd_pcm_sw_params_set_xrun_mode(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_xrun_t val) __attribute__((deprecated));
snd_pcm_xrun_t snd_pcm_sw_params_get_xrun_mode(const snd_pcm_sw_params_t *params) __attribute__((deprecated));

int snd_pcm_sw_params_set_xfer_align(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, snd_pcm_uframes_t val) __attribute__((deprecated));
int snd_pcm_sw_params_get_xfer_align(const snd_pcm_sw_params_t *params, snd_pcm_uframes_t *val) __attribute__((deprecated));
int snd_pcm_sw_params_set_sleep_min(snd_pcm_t *pcm, snd_pcm_sw_params_t *params, unsigned int val) __attribute__((deprecated));
int snd_pcm_sw_params_get_sleep_min(const snd_pcm_sw_params_t *params, unsigned int *val) __attribute__((deprecated));


int snd_pcm_hw_params_get_tick_time(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir) __attribute__((deprecated));
int snd_pcm_hw_params_get_tick_time_min(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir) __attribute__((deprecated));
int snd_pcm_hw_params_get_tick_time_max(const snd_pcm_hw_params_t *params, unsigned int *val, int *dir) __attribute__((deprecated));
int snd_pcm_hw_params_test_tick_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir) __attribute__((deprecated));
int snd_pcm_hw_params_set_tick_time(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int val, int dir) __attribute__((deprecated));
int snd_pcm_hw_params_set_tick_time_min(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir) __attribute__((deprecated));
int snd_pcm_hw_params_set_tick_time_max(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir) __attribute__((deprecated));
int snd_pcm_hw_params_set_tick_time_minmax(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *min, int *mindir, unsigned int *max, int *maxdir) __attribute__((deprecated));
int snd_pcm_hw_params_set_tick_time_near(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir) __attribute__((deprecated));
int snd_pcm_hw_params_set_tick_time_first(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir) __attribute__((deprecated));
int snd_pcm_hw_params_set_tick_time_last(snd_pcm_t *pcm, snd_pcm_hw_params_t *params, unsigned int *val, int *dir) __attribute__((deprecated));
# 50 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/rawmidi.h" 1 3
# 45 "/usr/include/alsa/rawmidi.h" 3
typedef struct _snd_rawmidi_info snd_rawmidi_info_t;

typedef struct _snd_rawmidi_params snd_rawmidi_params_t;

typedef struct _snd_rawmidi_status snd_rawmidi_status_t;


typedef enum _snd_rawmidi_stream {

 SND_RAWMIDI_STREAM_OUTPUT = 0,

 SND_RAWMIDI_STREAM_INPUT,
 SND_RAWMIDI_STREAM_LAST = SND_RAWMIDI_STREAM_INPUT
} snd_rawmidi_stream_t;
# 68 "/usr/include/alsa/rawmidi.h" 3
typedef struct _snd_rawmidi snd_rawmidi_t;


typedef enum _snd_rawmidi_type {

 SND_RAWMIDI_TYPE_HW,

 SND_RAWMIDI_TYPE_SHM,

 SND_RAWMIDI_TYPE_INET,

 SND_RAWMIDI_TYPE_VIRTUAL
} snd_rawmidi_type_t;

int snd_rawmidi_open(snd_rawmidi_t **in_rmidi, snd_rawmidi_t **out_rmidi,
       const char *name, int mode);
int snd_rawmidi_open_lconf(snd_rawmidi_t **in_rmidi, snd_rawmidi_t **out_rmidi,
      const char *name, int mode, snd_config_t *lconf);
int snd_rawmidi_close(snd_rawmidi_t *rmidi);
int snd_rawmidi_poll_descriptors_count(snd_rawmidi_t *rmidi);
int snd_rawmidi_poll_descriptors(snd_rawmidi_t *rmidi, struct pollfd *pfds, unsigned int space);
int snd_rawmidi_poll_descriptors_revents(snd_rawmidi_t *rawmidi, struct pollfd *pfds, unsigned int nfds, unsigned short *revent);
int snd_rawmidi_nonblock(snd_rawmidi_t *rmidi, int nonblock);
size_t snd_rawmidi_info_sizeof(void);





int snd_rawmidi_info_malloc(snd_rawmidi_info_t **ptr);
void snd_rawmidi_info_free(snd_rawmidi_info_t *obj);
void snd_rawmidi_info_copy(snd_rawmidi_info_t *dst, const snd_rawmidi_info_t *src);
unsigned int snd_rawmidi_info_get_device(const snd_rawmidi_info_t *obj);
unsigned int snd_rawmidi_info_get_subdevice(const snd_rawmidi_info_t *obj);
snd_rawmidi_stream_t snd_rawmidi_info_get_stream(const snd_rawmidi_info_t *obj);
int snd_rawmidi_info_get_card(const snd_rawmidi_info_t *obj);
unsigned int snd_rawmidi_info_get_flags(const snd_rawmidi_info_t *obj);
const char *snd_rawmidi_info_get_id(const snd_rawmidi_info_t *obj);
const char *snd_rawmidi_info_get_name(const snd_rawmidi_info_t *obj);
const char *snd_rawmidi_info_get_subdevice_name(const snd_rawmidi_info_t *obj);
unsigned int snd_rawmidi_info_get_subdevices_count(const snd_rawmidi_info_t *obj);
unsigned int snd_rawmidi_info_get_subdevices_avail(const snd_rawmidi_info_t *obj);
void snd_rawmidi_info_set_device(snd_rawmidi_info_t *obj, unsigned int val);
void snd_rawmidi_info_set_subdevice(snd_rawmidi_info_t *obj, unsigned int val);
void snd_rawmidi_info_set_stream(snd_rawmidi_info_t *obj, snd_rawmidi_stream_t val);
int snd_rawmidi_info(snd_rawmidi_t *rmidi, snd_rawmidi_info_t * info);
size_t snd_rawmidi_params_sizeof(void);





int snd_rawmidi_params_malloc(snd_rawmidi_params_t **ptr);
void snd_rawmidi_params_free(snd_rawmidi_params_t *obj);
void snd_rawmidi_params_copy(snd_rawmidi_params_t *dst, const snd_rawmidi_params_t *src);
int snd_rawmidi_params_set_buffer_size(snd_rawmidi_t *rmidi, snd_rawmidi_params_t *params, size_t val);
size_t snd_rawmidi_params_get_buffer_size(const snd_rawmidi_params_t *params);
int snd_rawmidi_params_set_avail_min(snd_rawmidi_t *rmidi, snd_rawmidi_params_t *params, size_t val);
size_t snd_rawmidi_params_get_avail_min(const snd_rawmidi_params_t *params);
int snd_rawmidi_params_set_no_active_sensing(snd_rawmidi_t *rmidi, snd_rawmidi_params_t *params, int val);
int snd_rawmidi_params_get_no_active_sensing(const snd_rawmidi_params_t *params);
int snd_rawmidi_params(snd_rawmidi_t *rmidi, snd_rawmidi_params_t * params);
int snd_rawmidi_params_current(snd_rawmidi_t *rmidi, snd_rawmidi_params_t *params);
size_t snd_rawmidi_status_sizeof(void);





int snd_rawmidi_status_malloc(snd_rawmidi_status_t **ptr);
void snd_rawmidi_status_free(snd_rawmidi_status_t *obj);
void snd_rawmidi_status_copy(snd_rawmidi_status_t *dst, const snd_rawmidi_status_t *src);
void snd_rawmidi_status_get_tstamp(const snd_rawmidi_status_t *obj, snd_htimestamp_t *ptr);
size_t snd_rawmidi_status_get_avail(const snd_rawmidi_status_t *obj);
size_t snd_rawmidi_status_get_xruns(const snd_rawmidi_status_t *obj);
int snd_rawmidi_status(snd_rawmidi_t *rmidi, snd_rawmidi_status_t * status);
int snd_rawmidi_drain(snd_rawmidi_t *rmidi);
int snd_rawmidi_drop(snd_rawmidi_t *rmidi);
ssize_t snd_rawmidi_write(snd_rawmidi_t *rmidi, const void *buffer, size_t size);
ssize_t snd_rawmidi_read(snd_rawmidi_t *rmidi, void *buffer, size_t size);
const char *snd_rawmidi_name(snd_rawmidi_t *rmidi);
snd_rawmidi_type_t snd_rawmidi_type(snd_rawmidi_t *rmidi);
snd_rawmidi_stream_t snd_rawmidi_stream(snd_rawmidi_t *rawmidi);
# 51 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/timer.h" 1 3
# 47 "/usr/include/alsa/timer.h" 3
typedef struct _snd_timer_id snd_timer_id_t;

typedef struct _snd_timer_ginfo snd_timer_ginfo_t;

typedef struct _snd_timer_gparams snd_timer_gparams_t;

typedef struct _snd_timer_gstatus snd_timer_gstatus_t;

typedef struct _snd_timer_info snd_timer_info_t;

typedef struct _snd_timer_params snd_timer_params_t;

typedef struct _snd_timer_status snd_timer_status_t;

typedef enum _snd_timer_class {
 SND_TIMER_CLASS_NONE = -1,
 SND_TIMER_CLASS_SLAVE = 0,
 SND_TIMER_CLASS_GLOBAL,
 SND_TIMER_CLASS_CARD,
 SND_TIMER_CLASS_PCM,
 SND_TIMER_CLASS_LAST = SND_TIMER_CLASS_PCM
} snd_timer_class_t;


typedef enum _snd_timer_slave_class {
 SND_TIMER_SCLASS_NONE = 0,
 SND_TIMER_SCLASS_APPLICATION,
 SND_TIMER_SCLASS_SEQUENCER,
 SND_TIMER_SCLASS_OSS_SEQUENCER,
 SND_TIMER_SCLASS_LAST = SND_TIMER_SCLASS_OSS_SEQUENCER
} snd_timer_slave_class_t;


typedef enum _snd_timer_event {
 SND_TIMER_EVENT_RESOLUTION = 0,
 SND_TIMER_EVENT_TICK,
 SND_TIMER_EVENT_START,
 SND_TIMER_EVENT_STOP,
 SND_TIMER_EVENT_CONTINUE,
 SND_TIMER_EVENT_PAUSE,
 SND_TIMER_EVENT_EARLY,
 SND_TIMER_EVENT_SUSPEND,
 SND_TIMER_EVENT_RESUME,

 SND_TIMER_EVENT_MSTART = SND_TIMER_EVENT_START + 10,
 SND_TIMER_EVENT_MSTOP = SND_TIMER_EVENT_STOP + 10,
 SND_TIMER_EVENT_MCONTINUE = SND_TIMER_EVENT_CONTINUE + 10,
 SND_TIMER_EVENT_MPAUSE = SND_TIMER_EVENT_PAUSE + 10,
 SND_TIMER_EVENT_MSUSPEND = SND_TIMER_EVENT_SUSPEND + 10,
 SND_TIMER_EVENT_MRESUME = SND_TIMER_EVENT_RESUME + 10
} snd_timer_event_t;


typedef struct _snd_timer_read {
 unsigned int resolution;
        unsigned int lo;
} snd_timer_read_t;


typedef struct _snd_timer_tread {
 snd_timer_event_t event;
 snd_htimestamp_t tstamp;
 unsigned int val;
} snd_timer_tread_t;
# 125 "/usr/include/alsa/timer.h" 3
typedef enum _snd_timer_type {

 SND_TIMER_TYPE_HW = 0,

 SND_TIMER_TYPE_SHM,

 SND_TIMER_TYPE_INET
} snd_timer_type_t;


typedef struct _snd_timer_query snd_timer_query_t;

typedef struct _snd_timer snd_timer_t;


int snd_timer_query_open(snd_timer_query_t **handle, const char *name, int mode);
int snd_timer_query_open_lconf(snd_timer_query_t **handle, const char *name, int mode, snd_config_t *lconf);
int snd_timer_query_close(snd_timer_query_t *handle);
int snd_timer_query_next_device(snd_timer_query_t *handle, snd_timer_id_t *tid);
int snd_timer_query_info(snd_timer_query_t *handle, snd_timer_ginfo_t *info);
int snd_timer_query_params(snd_timer_query_t *handle, snd_timer_gparams_t *params);
int snd_timer_query_status(snd_timer_query_t *handle, snd_timer_gstatus_t *status);

int snd_timer_open(snd_timer_t **handle, const char *name, int mode);
int snd_timer_open_lconf(snd_timer_t **handle, const char *name, int mode, snd_config_t *lconf);
int snd_timer_close(snd_timer_t *handle);
int snd_async_add_timer_handler(snd_async_handler_t **handler, snd_timer_t *timer,
    snd_async_callback_t callback, void *private_data);
snd_timer_t *snd_async_handler_get_timer(snd_async_handler_t *handler);
int snd_timer_poll_descriptors_count(snd_timer_t *handle);
int snd_timer_poll_descriptors(snd_timer_t *handle, struct pollfd *pfds, unsigned int space);
int snd_timer_poll_descriptors_revents(snd_timer_t *timer, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
int snd_timer_info(snd_timer_t *handle, snd_timer_info_t *timer);
int snd_timer_params(snd_timer_t *handle, snd_timer_params_t *params);
int snd_timer_status(snd_timer_t *handle, snd_timer_status_t *status);
int snd_timer_start(snd_timer_t *handle);
int snd_timer_stop(snd_timer_t *handle);
int snd_timer_continue(snd_timer_t *handle);
ssize_t snd_timer_read(snd_timer_t *handle, void *buffer, size_t size);

size_t snd_timer_id_sizeof(void);


int snd_timer_id_malloc(snd_timer_id_t **ptr);
void snd_timer_id_free(snd_timer_id_t *obj);
void snd_timer_id_copy(snd_timer_id_t *dst, const snd_timer_id_t *src);

void snd_timer_id_set_class(snd_timer_id_t *id, int dev_class);
int snd_timer_id_get_class(snd_timer_id_t *id);
void snd_timer_id_set_sclass(snd_timer_id_t *id, int dev_sclass);
int snd_timer_id_get_sclass(snd_timer_id_t *id);
void snd_timer_id_set_card(snd_timer_id_t *id, int card);
int snd_timer_id_get_card(snd_timer_id_t *id);
void snd_timer_id_set_device(snd_timer_id_t *id, int device);
int snd_timer_id_get_device(snd_timer_id_t *id);
void snd_timer_id_set_subdevice(snd_timer_id_t *id, int subdevice);
int snd_timer_id_get_subdevice(snd_timer_id_t *id);

size_t snd_timer_ginfo_sizeof(void);


int snd_timer_ginfo_malloc(snd_timer_ginfo_t **ptr);
void snd_timer_ginfo_free(snd_timer_ginfo_t *obj);
void snd_timer_ginfo_copy(snd_timer_ginfo_t *dst, const snd_timer_ginfo_t *src);

int snd_timer_ginfo_set_tid(snd_timer_ginfo_t *obj, snd_timer_id_t *tid);
snd_timer_id_t *snd_timer_ginfo_get_tid(snd_timer_ginfo_t *obj);
unsigned int snd_timer_ginfo_get_flags(snd_timer_ginfo_t *obj);
int snd_timer_ginfo_get_card(snd_timer_ginfo_t *obj);
char *snd_timer_ginfo_get_id(snd_timer_ginfo_t *obj);
char *snd_timer_ginfo_get_name(snd_timer_ginfo_t *obj);
unsigned long snd_timer_ginfo_get_resolution(snd_timer_ginfo_t *obj);
unsigned long snd_timer_ginfo_get_resolution_min(snd_timer_ginfo_t *obj);
unsigned long snd_timer_ginfo_get_resolution_max(snd_timer_ginfo_t *obj);
unsigned int snd_timer_ginfo_get_clients(snd_timer_ginfo_t *obj);

size_t snd_timer_info_sizeof(void);


int snd_timer_info_malloc(snd_timer_info_t **ptr);
void snd_timer_info_free(snd_timer_info_t *obj);
void snd_timer_info_copy(snd_timer_info_t *dst, const snd_timer_info_t *src);

int snd_timer_info_is_slave(snd_timer_info_t * info);
int snd_timer_info_get_card(snd_timer_info_t * info);
const char *snd_timer_info_get_id(snd_timer_info_t * info);
const char *snd_timer_info_get_name(snd_timer_info_t * info);
long snd_timer_info_get_resolution(snd_timer_info_t * info);

size_t snd_timer_params_sizeof(void);


int snd_timer_params_malloc(snd_timer_params_t **ptr);
void snd_timer_params_free(snd_timer_params_t *obj);
void snd_timer_params_copy(snd_timer_params_t *dst, const snd_timer_params_t *src);

int snd_timer_params_set_auto_start(snd_timer_params_t * params, int auto_start);
int snd_timer_params_get_auto_start(snd_timer_params_t * params);
int snd_timer_params_set_exclusive(snd_timer_params_t * params, int exclusive);
int snd_timer_params_get_exclusive(snd_timer_params_t * params);
int snd_timer_params_set_early_event(snd_timer_params_t * params, int early_event);
int snd_timer_params_get_early_event(snd_timer_params_t * params);
void snd_timer_params_set_ticks(snd_timer_params_t * params, long lo);
long snd_timer_params_get_ticks(snd_timer_params_t * params);
void snd_timer_params_set_queue_size(snd_timer_params_t * params, long queue_size);
long snd_timer_params_get_queue_size(snd_timer_params_t * params);
void snd_timer_params_set_filter(snd_timer_params_t * params, unsigned int filter);
unsigned int snd_timer_params_get_filter(snd_timer_params_t * params);

size_t snd_timer_status_sizeof(void);


int snd_timer_status_malloc(snd_timer_status_t **ptr);
void snd_timer_status_free(snd_timer_status_t *obj);
void snd_timer_status_copy(snd_timer_status_t *dst, const snd_timer_status_t *src);

snd_htimestamp_t snd_timer_status_get_timestamp(snd_timer_status_t * status);
long snd_timer_status_get_resolution(snd_timer_status_t * status);
long snd_timer_status_get_lost(snd_timer_status_t * status);
long snd_timer_status_get_overrun(snd_timer_status_t * status);
long snd_timer_status_get_queue(snd_timer_status_t * status);


long snd_timer_info_get_ticks(snd_timer_info_t * info);
# 52 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/hwdep.h" 1 3
# 45 "/usr/include/alsa/hwdep.h" 3
typedef struct _snd_hwdep_info snd_hwdep_info_t;


typedef struct _snd_hwdep_dsp_status snd_hwdep_dsp_status_t;


typedef struct _snd_hwdep_dsp_image snd_hwdep_dsp_image_t;


typedef enum _snd_hwdep_iface {
 SND_HWDEP_IFACE_OPL2 = 0,
 SND_HWDEP_IFACE_OPL3,
 SND_HWDEP_IFACE_OPL4,
 SND_HWDEP_IFACE_SB16CSP,
 SND_HWDEP_IFACE_EMU10K1,
 SND_HWDEP_IFACE_YSS225,
 SND_HWDEP_IFACE_ICS2115,
 SND_HWDEP_IFACE_SSCAPE,
 SND_HWDEP_IFACE_VX,
 SND_HWDEP_IFACE_MIXART,
 SND_HWDEP_IFACE_USX2Y,
 SND_HWDEP_IFACE_EMUX_WAVETABLE,
 SND_HWDEP_IFACE_BLUETOOTH,
 SND_HWDEP_IFACE_USX2Y_PCM,
 SND_HWDEP_IFACE_PCXHR,
 SND_HWDEP_IFACE_SB_RC,

 SND_HWDEP_IFACE_LAST = SND_HWDEP_IFACE_SB_RC
} snd_hwdep_iface_t;
# 85 "/usr/include/alsa/hwdep.h" 3
typedef enum _snd_hwdep_type {

 SND_HWDEP_TYPE_HW,

 SND_HWDEP_TYPE_SHM,

 SND_HWDEP_TYPE_INET
} snd_hwdep_type_t;


typedef struct _snd_hwdep snd_hwdep_t;

int snd_hwdep_open(snd_hwdep_t **hwdep, const char *name, int mode);
int snd_hwdep_close(snd_hwdep_t *hwdep);
int snd_hwdep_poll_descriptors(snd_hwdep_t *hwdep, struct pollfd *pfds, unsigned int space);
int snd_hwdep_poll_descriptors_revents(snd_hwdep_t *hwdep, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
int snd_hwdep_nonblock(snd_hwdep_t *hwdep, int nonblock);
int snd_hwdep_info(snd_hwdep_t *hwdep, snd_hwdep_info_t * info);
int snd_hwdep_dsp_status(snd_hwdep_t *hwdep, snd_hwdep_dsp_status_t *status);
int snd_hwdep_dsp_load(snd_hwdep_t *hwdep, snd_hwdep_dsp_image_t *block);
int snd_hwdep_ioctl(snd_hwdep_t *hwdep, unsigned int request, void * arg);
ssize_t snd_hwdep_write(snd_hwdep_t *hwdep, const void *buffer, size_t size);
ssize_t snd_hwdep_read(snd_hwdep_t *hwdep, void *buffer, size_t size);

size_t snd_hwdep_info_sizeof(void);


int snd_hwdep_info_malloc(snd_hwdep_info_t **ptr);
void snd_hwdep_info_free(snd_hwdep_info_t *obj);
void snd_hwdep_info_copy(snd_hwdep_info_t *dst, const snd_hwdep_info_t *src);

unsigned int snd_hwdep_info_get_device(const snd_hwdep_info_t *obj);
int snd_hwdep_info_get_card(const snd_hwdep_info_t *obj);
const char *snd_hwdep_info_get_id(const snd_hwdep_info_t *obj);
const char *snd_hwdep_info_get_name(const snd_hwdep_info_t *obj);
snd_hwdep_iface_t snd_hwdep_info_get_iface(const snd_hwdep_info_t *obj);
void snd_hwdep_info_set_device(snd_hwdep_info_t *obj, unsigned int val);

size_t snd_hwdep_dsp_status_sizeof(void);


int snd_hwdep_dsp_status_malloc(snd_hwdep_dsp_status_t **ptr);
void snd_hwdep_dsp_status_free(snd_hwdep_dsp_status_t *obj);
void snd_hwdep_dsp_status_copy(snd_hwdep_dsp_status_t *dst, const snd_hwdep_dsp_status_t *src);

unsigned int snd_hwdep_dsp_status_get_version(const snd_hwdep_dsp_status_t *obj);
const char *snd_hwdep_dsp_status_get_id(const snd_hwdep_dsp_status_t *obj);
unsigned int snd_hwdep_dsp_status_get_num_dsps(const snd_hwdep_dsp_status_t *obj);
unsigned int snd_hwdep_dsp_status_get_dsp_loaded(const snd_hwdep_dsp_status_t *obj);
unsigned int snd_hwdep_dsp_status_get_chip_ready(const snd_hwdep_dsp_status_t *obj);

size_t snd_hwdep_dsp_image_sizeof(void);


int snd_hwdep_dsp_image_malloc(snd_hwdep_dsp_image_t **ptr);
void snd_hwdep_dsp_image_free(snd_hwdep_dsp_image_t *obj);
void snd_hwdep_dsp_image_copy(snd_hwdep_dsp_image_t *dst, const snd_hwdep_dsp_image_t *src);

unsigned int snd_hwdep_dsp_image_get_index(const snd_hwdep_dsp_image_t *obj);
const char *snd_hwdep_dsp_image_get_name(const snd_hwdep_dsp_image_t *obj);
const void *snd_hwdep_dsp_image_get_image(const snd_hwdep_dsp_image_t *obj);
size_t snd_hwdep_dsp_image_get_length(const snd_hwdep_dsp_image_t *obj);

void snd_hwdep_dsp_image_set_index(snd_hwdep_dsp_image_t *obj, unsigned int _index);
void snd_hwdep_dsp_image_set_name(snd_hwdep_dsp_image_t *obj, const char *name);
void snd_hwdep_dsp_image_set_image(snd_hwdep_dsp_image_t *obj, void *buffer);
void snd_hwdep_dsp_image_set_length(snd_hwdep_dsp_image_t *obj, size_t length);
# 53 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/control.h" 1 3
# 46 "/usr/include/alsa/control.h" 3
typedef struct snd_aes_iec958 {
 unsigned char status[24];
 unsigned char subcode[147];
 unsigned char pad;
 unsigned char dig_subframe[4];
} snd_aes_iec958_t;


typedef struct _snd_ctl_card_info snd_ctl_card_info_t;


typedef struct _snd_ctl_elem_id snd_ctl_elem_id_t;


typedef struct _snd_ctl_elem_list snd_ctl_elem_list_t;


typedef struct _snd_ctl_elem_info snd_ctl_elem_info_t;


typedef struct _snd_ctl_elem_value snd_ctl_elem_value_t;


typedef struct _snd_ctl_event snd_ctl_event_t;


typedef enum _snd_ctl_elem_type {

 SND_CTL_ELEM_TYPE_NONE = 0,

 SND_CTL_ELEM_TYPE_BOOLEAN,

 SND_CTL_ELEM_TYPE_INTEGER,

 SND_CTL_ELEM_TYPE_ENUMERATED,

 SND_CTL_ELEM_TYPE_BYTES,

 SND_CTL_ELEM_TYPE_IEC958,

 SND_CTL_ELEM_TYPE_INTEGER64,
 SND_CTL_ELEM_TYPE_LAST = SND_CTL_ELEM_TYPE_INTEGER64
} snd_ctl_elem_type_t;


typedef enum _snd_ctl_elem_iface {

 SND_CTL_ELEM_IFACE_CARD = 0,

 SND_CTL_ELEM_IFACE_HWDEP,

 SND_CTL_ELEM_IFACE_MIXER,

 SND_CTL_ELEM_IFACE_PCM,

 SND_CTL_ELEM_IFACE_RAWMIDI,

 SND_CTL_ELEM_IFACE_TIMER,

 SND_CTL_ELEM_IFACE_SEQUENCER,
 SND_CTL_ELEM_IFACE_LAST = SND_CTL_ELEM_IFACE_SEQUENCER
} snd_ctl_elem_iface_t;


typedef enum _snd_ctl_event_type {

 SND_CTL_EVENT_ELEM = 0,
 SND_CTL_EVENT_LAST = SND_CTL_EVENT_ELEM
}snd_ctl_event_type_t;
# 182 "/usr/include/alsa/control.h" 3
typedef enum _snd_ctl_type {

 SND_CTL_TYPE_HW,

 SND_CTL_TYPE_SHM,

 SND_CTL_TYPE_INET,

 SND_CTL_TYPE_EXT
} snd_ctl_type_t;
# 203 "/usr/include/alsa/control.h" 3
typedef struct _snd_ctl snd_ctl_t;





typedef struct _snd_sctl snd_sctl_t;

int snd_card_load(int card);
int snd_card_next(int *card);
int snd_card_get_index(const char *name);
int snd_card_get_name(int card, char **name);
int snd_card_get_longname(int card, char **name);

int snd_device_name_hint(int card, const char *iface, void ***hints);
int snd_device_name_free_hint(void **hints);
char *snd_device_name_get_hint(const void *hint, const char *id);

int snd_ctl_open(snd_ctl_t **ctl, const char *name, int mode);
int snd_ctl_open_lconf(snd_ctl_t **ctl, const char *name, int mode, snd_config_t *lconf);
int snd_ctl_close(snd_ctl_t *ctl);
int snd_ctl_nonblock(snd_ctl_t *ctl, int nonblock);
int snd_async_add_ctl_handler(snd_async_handler_t **handler, snd_ctl_t *ctl,
         snd_async_callback_t callback, void *private_data);
snd_ctl_t *snd_async_handler_get_ctl(snd_async_handler_t *handler);
int snd_ctl_poll_descriptors_count(snd_ctl_t *ctl);
int snd_ctl_poll_descriptors(snd_ctl_t *ctl, struct pollfd *pfds, unsigned int space);
int snd_ctl_poll_descriptors_revents(snd_ctl_t *ctl, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
int snd_ctl_subscribe_events(snd_ctl_t *ctl, int subscribe);
int snd_ctl_card_info(snd_ctl_t *ctl, snd_ctl_card_info_t *info);
int snd_ctl_elem_list(snd_ctl_t *ctl, snd_ctl_elem_list_t *list);
int snd_ctl_elem_info(snd_ctl_t *ctl, snd_ctl_elem_info_t *info);
int snd_ctl_elem_read(snd_ctl_t *ctl, snd_ctl_elem_value_t *value);
int snd_ctl_elem_write(snd_ctl_t *ctl, snd_ctl_elem_value_t *value);
int snd_ctl_elem_lock(snd_ctl_t *ctl, snd_ctl_elem_id_t *id);
int snd_ctl_elem_unlock(snd_ctl_t *ctl, snd_ctl_elem_id_t *id);
int snd_ctl_elem_tlv_read(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
     unsigned int *tlv, unsigned int tlv_size);
int snd_ctl_elem_tlv_write(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
      const unsigned int *tlv);
int snd_ctl_elem_tlv_command(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
        const unsigned int *tlv);

int snd_ctl_hwdep_next_device(snd_ctl_t *ctl, int * device);
int snd_ctl_hwdep_info(snd_ctl_t *ctl, snd_hwdep_info_t * info);


int snd_ctl_pcm_next_device(snd_ctl_t *ctl, int *device);
int snd_ctl_pcm_info(snd_ctl_t *ctl, snd_pcm_info_t * info);
int snd_ctl_pcm_prefer_subdevice(snd_ctl_t *ctl, int subdev);


int snd_ctl_rawmidi_next_device(snd_ctl_t *ctl, int * device);
int snd_ctl_rawmidi_info(snd_ctl_t *ctl, snd_rawmidi_info_t * info);
int snd_ctl_rawmidi_prefer_subdevice(snd_ctl_t *ctl, int subdev);

int snd_ctl_set_power_state(snd_ctl_t *ctl, unsigned int state);
int snd_ctl_get_power_state(snd_ctl_t *ctl, unsigned int *state);

int snd_ctl_read(snd_ctl_t *ctl, snd_ctl_event_t *event);
int snd_ctl_wait(snd_ctl_t *ctl, int timeout);
const char *snd_ctl_name(snd_ctl_t *ctl);
snd_ctl_type_t snd_ctl_type(snd_ctl_t *ctl);

const char *snd_ctl_elem_type_name(snd_ctl_elem_type_t type);
const char *snd_ctl_elem_iface_name(snd_ctl_elem_iface_t iface);
const char *snd_ctl_event_type_name(snd_ctl_event_type_t type);

unsigned int snd_ctl_event_elem_get_mask(const snd_ctl_event_t *obj);
unsigned int snd_ctl_event_elem_get_numid(const snd_ctl_event_t *obj);
void snd_ctl_event_elem_get_id(const snd_ctl_event_t *obj, snd_ctl_elem_id_t *ptr);
snd_ctl_elem_iface_t snd_ctl_event_elem_get_interface(const snd_ctl_event_t *obj);
unsigned int snd_ctl_event_elem_get_device(const snd_ctl_event_t *obj);
unsigned int snd_ctl_event_elem_get_subdevice(const snd_ctl_event_t *obj);
const char *snd_ctl_event_elem_get_name(const snd_ctl_event_t *obj);
unsigned int snd_ctl_event_elem_get_index(const snd_ctl_event_t *obj);

int snd_ctl_elem_list_alloc_space(snd_ctl_elem_list_t *obj, unsigned int entries);
void snd_ctl_elem_list_free_space(snd_ctl_elem_list_t *obj);

size_t snd_ctl_elem_id_sizeof(void);





int snd_ctl_elem_id_malloc(snd_ctl_elem_id_t **ptr);
void snd_ctl_elem_id_free(snd_ctl_elem_id_t *obj);
void snd_ctl_elem_id_clear(snd_ctl_elem_id_t *obj);
void snd_ctl_elem_id_copy(snd_ctl_elem_id_t *dst, const snd_ctl_elem_id_t *src);
unsigned int snd_ctl_elem_id_get_numid(const snd_ctl_elem_id_t *obj);
snd_ctl_elem_iface_t snd_ctl_elem_id_get_interface(const snd_ctl_elem_id_t *obj);
unsigned int snd_ctl_elem_id_get_device(const snd_ctl_elem_id_t *obj);
unsigned int snd_ctl_elem_id_get_subdevice(const snd_ctl_elem_id_t *obj);
const char *snd_ctl_elem_id_get_name(const snd_ctl_elem_id_t *obj);
unsigned int snd_ctl_elem_id_get_index(const snd_ctl_elem_id_t *obj);
void snd_ctl_elem_id_set_numid(snd_ctl_elem_id_t *obj, unsigned int val);
void snd_ctl_elem_id_set_interface(snd_ctl_elem_id_t *obj, snd_ctl_elem_iface_t val);
void snd_ctl_elem_id_set_device(snd_ctl_elem_id_t *obj, unsigned int val);
void snd_ctl_elem_id_set_subdevice(snd_ctl_elem_id_t *obj, unsigned int val);
void snd_ctl_elem_id_set_name(snd_ctl_elem_id_t *obj, const char *val);
void snd_ctl_elem_id_set_index(snd_ctl_elem_id_t *obj, unsigned int val);

size_t snd_ctl_card_info_sizeof(void);





int snd_ctl_card_info_malloc(snd_ctl_card_info_t **ptr);
void snd_ctl_card_info_free(snd_ctl_card_info_t *obj);
void snd_ctl_card_info_clear(snd_ctl_card_info_t *obj);
void snd_ctl_card_info_copy(snd_ctl_card_info_t *dst, const snd_ctl_card_info_t *src);
int snd_ctl_card_info_get_card(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_id(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_driver(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_name(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_longname(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_mixername(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_components(const snd_ctl_card_info_t *obj);

size_t snd_ctl_event_sizeof(void);





int snd_ctl_event_malloc(snd_ctl_event_t **ptr);
void snd_ctl_event_free(snd_ctl_event_t *obj);
void snd_ctl_event_clear(snd_ctl_event_t *obj);
void snd_ctl_event_copy(snd_ctl_event_t *dst, const snd_ctl_event_t *src);
snd_ctl_event_type_t snd_ctl_event_get_type(const snd_ctl_event_t *obj);

size_t snd_ctl_elem_list_sizeof(void);





int snd_ctl_elem_list_malloc(snd_ctl_elem_list_t **ptr);
void snd_ctl_elem_list_free(snd_ctl_elem_list_t *obj);
void snd_ctl_elem_list_clear(snd_ctl_elem_list_t *obj);
void snd_ctl_elem_list_copy(snd_ctl_elem_list_t *dst, const snd_ctl_elem_list_t *src);
void snd_ctl_elem_list_set_offset(snd_ctl_elem_list_t *obj, unsigned int val);
unsigned int snd_ctl_elem_list_get_used(const snd_ctl_elem_list_t *obj);
unsigned int snd_ctl_elem_list_get_count(const snd_ctl_elem_list_t *obj);
void snd_ctl_elem_list_get_id(const snd_ctl_elem_list_t *obj, unsigned int idx, snd_ctl_elem_id_t *ptr);
unsigned int snd_ctl_elem_list_get_numid(const snd_ctl_elem_list_t *obj, unsigned int idx);
snd_ctl_elem_iface_t snd_ctl_elem_list_get_interface(const snd_ctl_elem_list_t *obj, unsigned int idx);
unsigned int snd_ctl_elem_list_get_device(const snd_ctl_elem_list_t *obj, unsigned int idx);
unsigned int snd_ctl_elem_list_get_subdevice(const snd_ctl_elem_list_t *obj, unsigned int idx);
const char *snd_ctl_elem_list_get_name(const snd_ctl_elem_list_t *obj, unsigned int idx);
unsigned int snd_ctl_elem_list_get_index(const snd_ctl_elem_list_t *obj, unsigned int idx);

size_t snd_ctl_elem_info_sizeof(void);





int snd_ctl_elem_info_malloc(snd_ctl_elem_info_t **ptr);
void snd_ctl_elem_info_free(snd_ctl_elem_info_t *obj);
void snd_ctl_elem_info_clear(snd_ctl_elem_info_t *obj);
void snd_ctl_elem_info_copy(snd_ctl_elem_info_t *dst, const snd_ctl_elem_info_t *src);
snd_ctl_elem_type_t snd_ctl_elem_info_get_type(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_readable(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_writable(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_volatile(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_inactive(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_locked(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_tlv_readable(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_tlv_writable(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_tlv_commandable(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_owner(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_user(const snd_ctl_elem_info_t *obj);
pid_t snd_ctl_elem_info_get_owner(const snd_ctl_elem_info_t *obj);
unsigned int snd_ctl_elem_info_get_count(const snd_ctl_elem_info_t *obj);
long snd_ctl_elem_info_get_min(const snd_ctl_elem_info_t *obj);
long snd_ctl_elem_info_get_max(const snd_ctl_elem_info_t *obj);
long snd_ctl_elem_info_get_step(const snd_ctl_elem_info_t *obj);
long long snd_ctl_elem_info_get_min64(const snd_ctl_elem_info_t *obj);
long long snd_ctl_elem_info_get_max64(const snd_ctl_elem_info_t *obj);
long long snd_ctl_elem_info_get_step64(const snd_ctl_elem_info_t *obj);
unsigned int snd_ctl_elem_info_get_items(const snd_ctl_elem_info_t *obj);
void snd_ctl_elem_info_set_item(snd_ctl_elem_info_t *obj, unsigned int val);
const char *snd_ctl_elem_info_get_item_name(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_get_dimensions(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_get_dimension(const snd_ctl_elem_info_t *obj, unsigned int idx);
void snd_ctl_elem_info_get_id(const snd_ctl_elem_info_t *obj, snd_ctl_elem_id_t *ptr);
unsigned int snd_ctl_elem_info_get_numid(const snd_ctl_elem_info_t *obj);
snd_ctl_elem_iface_t snd_ctl_elem_info_get_interface(const snd_ctl_elem_info_t *obj);
unsigned int snd_ctl_elem_info_get_device(const snd_ctl_elem_info_t *obj);
unsigned int snd_ctl_elem_info_get_subdevice(const snd_ctl_elem_info_t *obj);
const char *snd_ctl_elem_info_get_name(const snd_ctl_elem_info_t *obj);
unsigned int snd_ctl_elem_info_get_index(const snd_ctl_elem_info_t *obj);
void snd_ctl_elem_info_set_id(snd_ctl_elem_info_t *obj, const snd_ctl_elem_id_t *ptr);
void snd_ctl_elem_info_set_numid(snd_ctl_elem_info_t *obj, unsigned int val);
void snd_ctl_elem_info_set_interface(snd_ctl_elem_info_t *obj, snd_ctl_elem_iface_t val);
void snd_ctl_elem_info_set_device(snd_ctl_elem_info_t *obj, unsigned int val);
void snd_ctl_elem_info_set_subdevice(snd_ctl_elem_info_t *obj, unsigned int val);
void snd_ctl_elem_info_set_name(snd_ctl_elem_info_t *obj, const char *val);
void snd_ctl_elem_info_set_index(snd_ctl_elem_info_t *obj, unsigned int val);

int snd_ctl_elem_add_integer(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id, unsigned int count, long imin, long imax, long istep);
int snd_ctl_elem_add_integer64(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id, unsigned int count, long long imin, long long imax, long long istep);
int snd_ctl_elem_add_boolean(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id, unsigned int count);
int snd_ctl_elem_add_iec958(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id);
int snd_ctl_elem_remove(snd_ctl_t *ctl, snd_ctl_elem_id_t *id);

size_t snd_ctl_elem_value_sizeof(void);





int snd_ctl_elem_value_malloc(snd_ctl_elem_value_t **ptr);
void snd_ctl_elem_value_free(snd_ctl_elem_value_t *obj);
void snd_ctl_elem_value_clear(snd_ctl_elem_value_t *obj);
void snd_ctl_elem_value_copy(snd_ctl_elem_value_t *dst, const snd_ctl_elem_value_t *src);
void snd_ctl_elem_value_get_id(const snd_ctl_elem_value_t *obj, snd_ctl_elem_id_t *ptr);
unsigned int snd_ctl_elem_value_get_numid(const snd_ctl_elem_value_t *obj);
snd_ctl_elem_iface_t snd_ctl_elem_value_get_interface(const snd_ctl_elem_value_t *obj);
unsigned int snd_ctl_elem_value_get_device(const snd_ctl_elem_value_t *obj);
unsigned int snd_ctl_elem_value_get_subdevice(const snd_ctl_elem_value_t *obj);
const char *snd_ctl_elem_value_get_name(const snd_ctl_elem_value_t *obj);
unsigned int snd_ctl_elem_value_get_index(const snd_ctl_elem_value_t *obj);
void snd_ctl_elem_value_set_id(snd_ctl_elem_value_t *obj, const snd_ctl_elem_id_t *ptr);
void snd_ctl_elem_value_set_numid(snd_ctl_elem_value_t *obj, unsigned int val);
void snd_ctl_elem_value_set_interface(snd_ctl_elem_value_t *obj, snd_ctl_elem_iface_t val);
void snd_ctl_elem_value_set_device(snd_ctl_elem_value_t *obj, unsigned int val);
void snd_ctl_elem_value_set_subdevice(snd_ctl_elem_value_t *obj, unsigned int val);
void snd_ctl_elem_value_set_name(snd_ctl_elem_value_t *obj, const char *val);
void snd_ctl_elem_value_set_index(snd_ctl_elem_value_t *obj, unsigned int val);
int snd_ctl_elem_value_get_boolean(const snd_ctl_elem_value_t *obj, unsigned int idx);
long snd_ctl_elem_value_get_integer(const snd_ctl_elem_value_t *obj, unsigned int idx);
long long snd_ctl_elem_value_get_integer64(const snd_ctl_elem_value_t *obj, unsigned int idx);
unsigned int snd_ctl_elem_value_get_enumerated(const snd_ctl_elem_value_t *obj, unsigned int idx);
unsigned char snd_ctl_elem_value_get_byte(const snd_ctl_elem_value_t *obj, unsigned int idx);
void snd_ctl_elem_value_set_boolean(snd_ctl_elem_value_t *obj, unsigned int idx, long val);
void snd_ctl_elem_value_set_integer(snd_ctl_elem_value_t *obj, unsigned int idx, long val);
void snd_ctl_elem_value_set_integer64(snd_ctl_elem_value_t *obj, unsigned int idx, long long val);
void snd_ctl_elem_value_set_enumerated(snd_ctl_elem_value_t *obj, unsigned int idx, unsigned int val);
void snd_ctl_elem_value_set_byte(snd_ctl_elem_value_t *obj, unsigned int idx, unsigned char val);
void snd_ctl_elem_set_bytes(snd_ctl_elem_value_t *obj, void *data, size_t size);
const void * snd_ctl_elem_value_get_bytes(const snd_ctl_elem_value_t *obj);
void snd_ctl_elem_value_get_iec958(const snd_ctl_elem_value_t *obj, snd_aes_iec958_t *ptr);
void snd_ctl_elem_value_set_iec958(snd_ctl_elem_value_t *obj, const snd_aes_iec958_t *ptr);

int snd_tlv_parse_dB_info(unsigned int *tlv, unsigned int tlv_size,
     unsigned int **db_tlvp);
int snd_tlv_get_dB_range(unsigned int *tlv, long rangemin, long rangemax,
    long *min, long *max);
int snd_tlv_convert_to_dB(unsigned int *tlv, long rangemin, long rangemax,
     long volume, long *db_gain);
int snd_tlv_convert_from_dB(unsigned int *tlv, long rangemin, long rangemax,
       long db_gain, long *value, int xdir);
int snd_ctl_get_dB_range(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
    long *min, long *max);
int snd_ctl_convert_to_dB(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
     long volume, long *db_gain);
int snd_ctl_convert_from_dB(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
       long db_gain, long *value, int xdir);
# 475 "/usr/include/alsa/control.h" 3
typedef struct _snd_hctl_elem snd_hctl_elem_t;


typedef struct _snd_hctl snd_hctl_t;







typedef int (*snd_hctl_compare_t)(const snd_hctl_elem_t *e1,
      const snd_hctl_elem_t *e2);
int snd_hctl_compare_fast(const snd_hctl_elem_t *c1,
     const snd_hctl_elem_t *c2);







typedef int (*snd_hctl_callback_t)(snd_hctl_t *hctl,
       unsigned int mask,
       snd_hctl_elem_t *elem);






typedef int (*snd_hctl_elem_callback_t)(snd_hctl_elem_t *elem,
     unsigned int mask);

int snd_hctl_open(snd_hctl_t **hctl, const char *name, int mode);
int snd_hctl_open_ctl(snd_hctl_t **hctlp, snd_ctl_t *ctl);
int snd_hctl_close(snd_hctl_t *hctl);
int snd_hctl_nonblock(snd_hctl_t *hctl, int nonblock);
int snd_hctl_poll_descriptors_count(snd_hctl_t *hctl);
int snd_hctl_poll_descriptors(snd_hctl_t *hctl, struct pollfd *pfds, unsigned int space);
int snd_hctl_poll_descriptors_revents(snd_hctl_t *ctl, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
unsigned int snd_hctl_get_count(snd_hctl_t *hctl);
int snd_hctl_set_compare(snd_hctl_t *hctl, snd_hctl_compare_t hsort);
snd_hctl_elem_t *snd_hctl_first_elem(snd_hctl_t *hctl);
snd_hctl_elem_t *snd_hctl_last_elem(snd_hctl_t *hctl);
snd_hctl_elem_t *snd_hctl_find_elem(snd_hctl_t *hctl, const snd_ctl_elem_id_t *id);
void snd_hctl_set_callback(snd_hctl_t *hctl, snd_hctl_callback_t callback);
void snd_hctl_set_callback_private(snd_hctl_t *hctl, void *data);
void *snd_hctl_get_callback_private(snd_hctl_t *hctl);
int snd_hctl_load(snd_hctl_t *hctl);
int snd_hctl_free(snd_hctl_t *hctl);
int snd_hctl_handle_events(snd_hctl_t *hctl);
const char *snd_hctl_name(snd_hctl_t *hctl);
int snd_hctl_wait(snd_hctl_t *hctl, int timeout);
snd_ctl_t *snd_hctl_ctl(snd_hctl_t *hctl);

snd_hctl_elem_t *snd_hctl_elem_next(snd_hctl_elem_t *elem);
snd_hctl_elem_t *snd_hctl_elem_prev(snd_hctl_elem_t *elem);
int snd_hctl_elem_info(snd_hctl_elem_t *elem, snd_ctl_elem_info_t * info);
int snd_hctl_elem_read(snd_hctl_elem_t *elem, snd_ctl_elem_value_t * value);
int snd_hctl_elem_write(snd_hctl_elem_t *elem, snd_ctl_elem_value_t * value);
int snd_hctl_elem_tlv_read(snd_hctl_elem_t *elem, unsigned int *tlv, unsigned int tlv_size);
int snd_hctl_elem_tlv_write(snd_hctl_elem_t *elem, const unsigned int *tlv);
int snd_hctl_elem_tlv_command(snd_hctl_elem_t *elem, const unsigned int *tlv);

snd_hctl_t *snd_hctl_elem_get_hctl(snd_hctl_elem_t *elem);

void snd_hctl_elem_get_id(const snd_hctl_elem_t *obj, snd_ctl_elem_id_t *ptr);
unsigned int snd_hctl_elem_get_numid(const snd_hctl_elem_t *obj);
snd_ctl_elem_iface_t snd_hctl_elem_get_interface(const snd_hctl_elem_t *obj);
unsigned int snd_hctl_elem_get_device(const snd_hctl_elem_t *obj);
unsigned int snd_hctl_elem_get_subdevice(const snd_hctl_elem_t *obj);
const char *snd_hctl_elem_get_name(const snd_hctl_elem_t *obj);
unsigned int snd_hctl_elem_get_index(const snd_hctl_elem_t *obj);
void snd_hctl_elem_set_callback(snd_hctl_elem_t *obj, snd_hctl_elem_callback_t val);
void * snd_hctl_elem_get_callback_private(const snd_hctl_elem_t *obj);
void snd_hctl_elem_set_callback_private(snd_hctl_elem_t *obj, void * val);
# 564 "/usr/include/alsa/control.h" 3
int snd_sctl_build(snd_sctl_t **ctl, snd_ctl_t *handle, snd_config_t *config,
     snd_config_t *private_data, int mode);
int snd_sctl_free(snd_sctl_t *handle);
int snd_sctl_install(snd_sctl_t *handle);
int snd_sctl_remove(snd_sctl_t *handle);
# 54 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/mixer.h" 1 3
# 42 "/usr/include/alsa/mixer.h" 3
typedef struct _snd_mixer snd_mixer_t;

typedef struct _snd_mixer_class snd_mixer_class_t;

typedef struct _snd_mixer_elem snd_mixer_elem_t;
# 55 "/usr/include/alsa/mixer.h" 3
typedef int (*snd_mixer_callback_t)(snd_mixer_t *ctl,
        unsigned int mask,
        snd_mixer_elem_t *elem);







typedef int (*snd_mixer_elem_callback_t)(snd_mixer_elem_t *elem,
      unsigned int mask);







typedef int (*snd_mixer_compare_t)(const snd_mixer_elem_t *e1,
       const snd_mixer_elem_t *e2);
# 85 "/usr/include/alsa/mixer.h" 3
typedef int (*snd_mixer_event_t)(snd_mixer_class_t *class_, unsigned int mask,
     snd_hctl_elem_t *helem, snd_mixer_elem_t *melem);



typedef enum _snd_mixer_elem_type {

 SND_MIXER_ELEM_SIMPLE,
 SND_MIXER_ELEM_LAST = SND_MIXER_ELEM_SIMPLE
} snd_mixer_elem_type_t;

int snd_mixer_open(snd_mixer_t **mixer, int mode);
int snd_mixer_close(snd_mixer_t *mixer);
snd_mixer_elem_t *snd_mixer_first_elem(snd_mixer_t *mixer);
snd_mixer_elem_t *snd_mixer_last_elem(snd_mixer_t *mixer);
int snd_mixer_handle_events(snd_mixer_t *mixer);
int snd_mixer_attach(snd_mixer_t *mixer, const char *name);
int snd_mixer_attach_hctl(snd_mixer_t *mixer, snd_hctl_t *hctl);
int snd_mixer_detach(snd_mixer_t *mixer, const char *name);
int snd_mixer_detach_hctl(snd_mixer_t *mixer, snd_hctl_t *hctl);
int snd_mixer_get_hctl(snd_mixer_t *mixer, const char *name, snd_hctl_t **hctl);
int snd_mixer_poll_descriptors_count(snd_mixer_t *mixer);
int snd_mixer_poll_descriptors(snd_mixer_t *mixer, struct pollfd *pfds, unsigned int space);
int snd_mixer_poll_descriptors_revents(snd_mixer_t *mixer, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
int snd_mixer_load(snd_mixer_t *mixer);
void snd_mixer_free(snd_mixer_t *mixer);
int snd_mixer_wait(snd_mixer_t *mixer, int timeout);
int snd_mixer_set_compare(snd_mixer_t *mixer, snd_mixer_compare_t msort);
void snd_mixer_set_callback(snd_mixer_t *obj, snd_mixer_callback_t val);
void * snd_mixer_get_callback_private(const snd_mixer_t *obj);
void snd_mixer_set_callback_private(snd_mixer_t *obj, void * val);
unsigned int snd_mixer_get_count(const snd_mixer_t *obj);
int snd_mixer_class_unregister(snd_mixer_class_t *clss);

snd_mixer_elem_t *snd_mixer_elem_next(snd_mixer_elem_t *elem);
snd_mixer_elem_t *snd_mixer_elem_prev(snd_mixer_elem_t *elem);
void snd_mixer_elem_set_callback(snd_mixer_elem_t *obj, snd_mixer_elem_callback_t val);
void * snd_mixer_elem_get_callback_private(const snd_mixer_elem_t *obj);
void snd_mixer_elem_set_callback_private(snd_mixer_elem_t *obj, void * val);
snd_mixer_elem_type_t snd_mixer_elem_get_type(const snd_mixer_elem_t *obj);

int snd_mixer_class_register(snd_mixer_class_t *class_, snd_mixer_t *mixer);
int snd_mixer_add_elem(snd_mixer_t *mixer, snd_mixer_elem_t *elem);
int snd_mixer_remove_elem(snd_mixer_t *mixer, snd_mixer_elem_t *elem);
int snd_mixer_elem_new(snd_mixer_elem_t **elem,
         snd_mixer_elem_type_t type,
         int compare_weight,
         void *private_data,
         void (*private_free)(snd_mixer_elem_t *elem));
int snd_mixer_elem_add(snd_mixer_elem_t *elem, snd_mixer_class_t *class_);
int snd_mixer_elem_remove(snd_mixer_elem_t *elem);
void snd_mixer_elem_free(snd_mixer_elem_t *elem);
int snd_mixer_elem_info(snd_mixer_elem_t *elem);
int snd_mixer_elem_value(snd_mixer_elem_t *elem);
int snd_mixer_elem_attach(snd_mixer_elem_t *melem, snd_hctl_elem_t *helem);
int snd_mixer_elem_detach(snd_mixer_elem_t *melem, snd_hctl_elem_t *helem);
int snd_mixer_elem_empty(snd_mixer_elem_t *melem);
void *snd_mixer_elem_get_private(const snd_mixer_elem_t *melem);

size_t snd_mixer_class_sizeof(void);





int snd_mixer_class_malloc(snd_mixer_class_t **ptr);
void snd_mixer_class_free(snd_mixer_class_t *obj);
void snd_mixer_class_copy(snd_mixer_class_t *dst, const snd_mixer_class_t *src);
snd_mixer_t *snd_mixer_class_get_mixer(const snd_mixer_class_t *class_);
snd_mixer_event_t snd_mixer_class_get_event(const snd_mixer_class_t *class_);
void *snd_mixer_class_get_private(const snd_mixer_class_t *class_);
snd_mixer_compare_t snd_mixer_class_get_compare(const snd_mixer_class_t *class_);
int snd_mixer_class_set_event(snd_mixer_class_t *class_, snd_mixer_event_t event);
int snd_mixer_class_set_private(snd_mixer_class_t *class_, void *private_data);
int snd_mixer_class_set_private_free(snd_mixer_class_t *class_, void (*private_free)(snd_mixer_class_t *class_));
int snd_mixer_class_set_compare(snd_mixer_class_t *class_, snd_mixer_compare_t compare);
# 172 "/usr/include/alsa/mixer.h" 3
typedef enum _snd_mixer_selem_channel_id {

 SND_MIXER_SCHN_UNKNOWN = -1,

 SND_MIXER_SCHN_FRONT_LEFT = 0,

 SND_MIXER_SCHN_FRONT_RIGHT,

 SND_MIXER_SCHN_REAR_LEFT,

 SND_MIXER_SCHN_REAR_RIGHT,

 SND_MIXER_SCHN_FRONT_CENTER,

 SND_MIXER_SCHN_WOOFER,

 SND_MIXER_SCHN_SIDE_LEFT,

 SND_MIXER_SCHN_SIDE_RIGHT,

 SND_MIXER_SCHN_REAR_CENTER,
 SND_MIXER_SCHN_LAST = 31,

 SND_MIXER_SCHN_MONO = SND_MIXER_SCHN_FRONT_LEFT
} snd_mixer_selem_channel_id_t;


enum snd_mixer_selem_regopt_abstract {

 SND_MIXER_SABSTRACT_NONE = 0,

 SND_MIXER_SABSTRACT_BASIC,
};


struct snd_mixer_selem_regopt {

 int ver;

 enum snd_mixer_selem_regopt_abstract abstract;

 const char *device;

 snd_pcm_t *playback_pcm;

 snd_pcm_t *capture_pcm;
};


typedef struct _snd_mixer_selem_id snd_mixer_selem_id_t;

const char *snd_mixer_selem_channel_name(snd_mixer_selem_channel_id_t channel);

int snd_mixer_selem_register(snd_mixer_t *mixer,
        struct snd_mixer_selem_regopt *options,
        snd_mixer_class_t **classp);
void snd_mixer_selem_get_id(snd_mixer_elem_t *element,
       snd_mixer_selem_id_t *id);
const char *snd_mixer_selem_get_name(snd_mixer_elem_t *elem);
unsigned int snd_mixer_selem_get_index(snd_mixer_elem_t *elem);
snd_mixer_elem_t *snd_mixer_find_selem(snd_mixer_t *mixer,
           const snd_mixer_selem_id_t *id);

int snd_mixer_selem_is_active(snd_mixer_elem_t *elem);
int snd_mixer_selem_is_playback_mono(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_playback_channel(snd_mixer_elem_t *obj, snd_mixer_selem_channel_id_t channel);
int snd_mixer_selem_is_capture_mono(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_channel(snd_mixer_elem_t *obj, snd_mixer_selem_channel_id_t channel);
int snd_mixer_selem_get_capture_group(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_common_volume(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_playback_volume(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_playback_volume_joined(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_volume(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_volume_joined(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_common_switch(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_playback_switch(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_playback_switch_joined(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_switch(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_switch_joined(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_switch_exclusive(snd_mixer_elem_t *elem);

int snd_mixer_selem_ask_playback_vol_dB(snd_mixer_elem_t *elem, long value, long *dBvalue);
int snd_mixer_selem_ask_capture_vol_dB(snd_mixer_elem_t *elem, long value, long *dBvalue);
int snd_mixer_selem_ask_playback_dB_vol(snd_mixer_elem_t *elem, long dBvalue, int dir, long *value);
int snd_mixer_selem_ask_capture_dB_vol(snd_mixer_elem_t *elem, long dBvalue, int dir, long *value);
int snd_mixer_selem_get_playback_volume(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long *value);
int snd_mixer_selem_get_capture_volume(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long *value);
int snd_mixer_selem_get_playback_dB(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long *value);
int snd_mixer_selem_get_capture_dB(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long *value);
int snd_mixer_selem_get_playback_switch(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, int *value);
int snd_mixer_selem_get_capture_switch(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, int *value);
int snd_mixer_selem_set_playback_volume(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long value);
int snd_mixer_selem_set_capture_volume(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long value);
int snd_mixer_selem_set_playback_dB(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long value, int dir);
int snd_mixer_selem_set_capture_dB(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long value, int dir);
int snd_mixer_selem_set_playback_volume_all(snd_mixer_elem_t *elem, long value);
int snd_mixer_selem_set_capture_volume_all(snd_mixer_elem_t *elem, long value);
int snd_mixer_selem_set_playback_dB_all(snd_mixer_elem_t *elem, long value, int dir);
int snd_mixer_selem_set_capture_dB_all(snd_mixer_elem_t *elem, long value, int dir);
int snd_mixer_selem_set_playback_switch(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, int value);
int snd_mixer_selem_set_capture_switch(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, int value);
int snd_mixer_selem_set_playback_switch_all(snd_mixer_elem_t *elem, int value);
int snd_mixer_selem_set_capture_switch_all(snd_mixer_elem_t *elem, int value);
int snd_mixer_selem_get_playback_volume_range(snd_mixer_elem_t *elem,
           long *min, long *max);
int snd_mixer_selem_get_playback_dB_range(snd_mixer_elem_t *elem,
       long *min, long *max);
int snd_mixer_selem_set_playback_volume_range(snd_mixer_elem_t *elem,
           long min, long max);
int snd_mixer_selem_get_capture_volume_range(snd_mixer_elem_t *elem,
          long *min, long *max);
int snd_mixer_selem_get_capture_dB_range(snd_mixer_elem_t *elem,
      long *min, long *max);
int snd_mixer_selem_set_capture_volume_range(snd_mixer_elem_t *elem,
          long min, long max);

int snd_mixer_selem_is_enumerated(snd_mixer_elem_t *elem);
int snd_mixer_selem_is_enum_playback(snd_mixer_elem_t *elem);
int snd_mixer_selem_is_enum_capture(snd_mixer_elem_t *elem);
int snd_mixer_selem_get_enum_items(snd_mixer_elem_t *elem);
int snd_mixer_selem_get_enum_item_name(snd_mixer_elem_t *elem, unsigned int idx, size_t maxlen, char *str);
int snd_mixer_selem_get_enum_item(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, unsigned int *idxp);
int snd_mixer_selem_set_enum_item(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, unsigned int idx);

size_t snd_mixer_selem_id_sizeof(void);





int snd_mixer_selem_id_malloc(snd_mixer_selem_id_t **ptr);
void snd_mixer_selem_id_free(snd_mixer_selem_id_t *obj);
void snd_mixer_selem_id_copy(snd_mixer_selem_id_t *dst, const snd_mixer_selem_id_t *src);
const char *snd_mixer_selem_id_get_name(const snd_mixer_selem_id_t *obj);
unsigned int snd_mixer_selem_id_get_index(const snd_mixer_selem_id_t *obj);
void snd_mixer_selem_id_set_name(snd_mixer_selem_id_t *obj, const char *val);
void snd_mixer_selem_id_set_index(snd_mixer_selem_id_t *obj, unsigned int val);
# 55 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/seq_event.h" 1 3
# 41 "/usr/include/alsa/seq_event.h" 3
typedef unsigned char snd_seq_event_type_t;


enum snd_seq_event_type {

 SND_SEQ_EVENT_SYSTEM = 0,

 SND_SEQ_EVENT_RESULT,


 SND_SEQ_EVENT_NOTE = 5,

 SND_SEQ_EVENT_NOTEON,

 SND_SEQ_EVENT_NOTEOFF,

 SND_SEQ_EVENT_KEYPRESS,


 SND_SEQ_EVENT_CONTROLLER = 10,

 SND_SEQ_EVENT_PGMCHANGE,

 SND_SEQ_EVENT_CHANPRESS,

 SND_SEQ_EVENT_PITCHBEND,

 SND_SEQ_EVENT_CONTROL14,

 SND_SEQ_EVENT_NONREGPARAM,

 SND_SEQ_EVENT_REGPARAM,


 SND_SEQ_EVENT_SONGPOS = 20,

 SND_SEQ_EVENT_SONGSEL,

 SND_SEQ_EVENT_QFRAME,

 SND_SEQ_EVENT_TIMESIGN,

 SND_SEQ_EVENT_KEYSIGN,


 SND_SEQ_EVENT_START = 30,

 SND_SEQ_EVENT_CONTINUE,

 SND_SEQ_EVENT_STOP,

 SND_SEQ_EVENT_SETPOS_TICK,

 SND_SEQ_EVENT_SETPOS_TIME,

 SND_SEQ_EVENT_TEMPO,

 SND_SEQ_EVENT_CLOCK,

 SND_SEQ_EVENT_TICK,

 SND_SEQ_EVENT_QUEUE_SKEW,

 SND_SEQ_EVENT_SYNC_POS,


 SND_SEQ_EVENT_TUNE_REQUEST = 40,

 SND_SEQ_EVENT_RESET,

 SND_SEQ_EVENT_SENSING,


 SND_SEQ_EVENT_ECHO = 50,

 SND_SEQ_EVENT_OSS,


 SND_SEQ_EVENT_CLIENT_START = 60,

 SND_SEQ_EVENT_CLIENT_EXIT,

 SND_SEQ_EVENT_CLIENT_CHANGE,

 SND_SEQ_EVENT_PORT_START,

 SND_SEQ_EVENT_PORT_EXIT,

 SND_SEQ_EVENT_PORT_CHANGE,


 SND_SEQ_EVENT_PORT_SUBSCRIBED,

 SND_SEQ_EVENT_PORT_UNSUBSCRIBED,


 SND_SEQ_EVENT_USR0 = 90,

 SND_SEQ_EVENT_USR1,

 SND_SEQ_EVENT_USR2,

 SND_SEQ_EVENT_USR3,

 SND_SEQ_EVENT_USR4,

 SND_SEQ_EVENT_USR5,

 SND_SEQ_EVENT_USR6,

 SND_SEQ_EVENT_USR7,

 SND_SEQ_EVENT_USR8,

 SND_SEQ_EVENT_USR9,


 SND_SEQ_EVENT_SYSEX = 130,

 SND_SEQ_EVENT_BOUNCE,

 SND_SEQ_EVENT_USR_VAR0 = 135,

 SND_SEQ_EVENT_USR_VAR1,

 SND_SEQ_EVENT_USR_VAR2,

 SND_SEQ_EVENT_USR_VAR3,

 SND_SEQ_EVENT_USR_VAR4,


 SND_SEQ_EVENT_NONE = 255
};



typedef struct snd_seq_addr {
 unsigned char client;
 unsigned char port;
} snd_seq_addr_t;


typedef struct snd_seq_connect {
 snd_seq_addr_t sender;
 snd_seq_addr_t dest;
} snd_seq_connect_t;



typedef struct snd_seq_real_time {
 unsigned int tv_sec;
 unsigned int tv_nsec;
} snd_seq_real_time_t;


typedef unsigned int snd_seq_tick_time_t;


typedef union snd_seq_timestamp {
 snd_seq_tick_time_t tick;
 struct snd_seq_real_time time;
} snd_seq_timestamp_t;
# 230 "/usr/include/alsa/seq_event.h" 3
typedef struct snd_seq_ev_note {
 unsigned char channel;
 unsigned char note;
 unsigned char velocity;
 unsigned char off_velocity;
 unsigned int duration;
} snd_seq_ev_note_t;


typedef struct snd_seq_ev_ctrl {
 unsigned char channel;
 unsigned char unused[3];
 unsigned int param;
 signed int value;
} snd_seq_ev_ctrl_t;


typedef struct snd_seq_ev_raw8 {
 unsigned char d[12];
} snd_seq_ev_raw8_t;


typedef struct snd_seq_ev_raw32 {
 unsigned int d[3];
} snd_seq_ev_raw32_t;


typedef struct snd_seq_ev_ext {
 unsigned int len;
 void *ptr;
} __attribute__((packed)) snd_seq_ev_ext_t;


typedef struct snd_seq_result {
 int event;
 int result;
} snd_seq_result_t;


typedef struct snd_seq_queue_skew {
 unsigned int value;
 unsigned int base;
} snd_seq_queue_skew_t;


typedef struct snd_seq_ev_queue_control {
 unsigned char queue;
 unsigned char unused[3];
 union {
  signed int value;
  snd_seq_timestamp_t time;
  unsigned int position;
  snd_seq_queue_skew_t skew;
  unsigned int d32[2];
  unsigned char d8[8];
 } param;
} snd_seq_ev_queue_control_t;



typedef struct snd_seq_event {
 snd_seq_event_type_t type;
 unsigned char flags;
 unsigned char tag;

 unsigned char queue;
 snd_seq_timestamp_t time;

 snd_seq_addr_t source;
 snd_seq_addr_t dest;

 union {
  snd_seq_ev_note_t note;
  snd_seq_ev_ctrl_t control;
  snd_seq_ev_raw8_t raw8;
  snd_seq_ev_raw32_t raw32;
  snd_seq_ev_ext_t ext;
  snd_seq_ev_queue_control_t queue;
  snd_seq_timestamp_t time;
  snd_seq_addr_t addr;
  snd_seq_connect_t connect;
  snd_seq_result_t result;
 } data;
} snd_seq_event_t;
# 56 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/seq.h" 1 3
# 47 "/usr/include/alsa/seq.h" 3
typedef struct _snd_seq snd_seq_t;
# 62 "/usr/include/alsa/seq.h" 3
typedef enum _snd_seq_type {
 SND_SEQ_TYPE_HW,
 SND_SEQ_TYPE_SHM,
 SND_SEQ_TYPE_INET
} snd_seq_type_t;
# 78 "/usr/include/alsa/seq.h" 3
int snd_seq_open(snd_seq_t **handle, const char *name, int streams, int mode);
int snd_seq_open_lconf(snd_seq_t **handle, const char *name, int streams, int mode, snd_config_t *lconf);
const char *snd_seq_name(snd_seq_t *seq);
snd_seq_type_t snd_seq_type(snd_seq_t *seq);
int snd_seq_close(snd_seq_t *handle);
int snd_seq_poll_descriptors_count(snd_seq_t *handle, short events);
int snd_seq_poll_descriptors(snd_seq_t *handle, struct pollfd *pfds, unsigned int space, short events);
int snd_seq_poll_descriptors_revents(snd_seq_t *seq, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
int snd_seq_nonblock(snd_seq_t *handle, int nonblock);
int snd_seq_client_id(snd_seq_t *handle);

size_t snd_seq_get_output_buffer_size(snd_seq_t *handle);
size_t snd_seq_get_input_buffer_size(snd_seq_t *handle);
int snd_seq_set_output_buffer_size(snd_seq_t *handle, size_t size);
int snd_seq_set_input_buffer_size(snd_seq_t *handle, size_t size);


typedef struct _snd_seq_system_info snd_seq_system_info_t;

size_t snd_seq_system_info_sizeof(void);



int snd_seq_system_info_malloc(snd_seq_system_info_t **ptr);
void snd_seq_system_info_free(snd_seq_system_info_t *ptr);
void snd_seq_system_info_copy(snd_seq_system_info_t *dst, const snd_seq_system_info_t *src);

int snd_seq_system_info_get_queues(const snd_seq_system_info_t *info);
int snd_seq_system_info_get_clients(const snd_seq_system_info_t *info);
int snd_seq_system_info_get_ports(const snd_seq_system_info_t *info);
int snd_seq_system_info_get_channels(const snd_seq_system_info_t *info);
int snd_seq_system_info_get_cur_clients(const snd_seq_system_info_t *info);
int snd_seq_system_info_get_cur_queues(const snd_seq_system_info_t *info);

int snd_seq_system_info(snd_seq_t *handle, snd_seq_system_info_t *info);
# 125 "/usr/include/alsa/seq.h" 3
typedef struct _snd_seq_client_info snd_seq_client_info_t;


typedef enum snd_seq_client_type {
 SND_SEQ_USER_CLIENT = 1,
 SND_SEQ_KERNEL_CLIENT = 2
} snd_seq_client_type_t;

size_t snd_seq_client_info_sizeof(void);



int snd_seq_client_info_malloc(snd_seq_client_info_t **ptr);
void snd_seq_client_info_free(snd_seq_client_info_t *ptr);
void snd_seq_client_info_copy(snd_seq_client_info_t *dst, const snd_seq_client_info_t *src);

int snd_seq_client_info_get_client(const snd_seq_client_info_t *info);
snd_seq_client_type_t snd_seq_client_info_get_type(const snd_seq_client_info_t *info);
const char *snd_seq_client_info_get_name(snd_seq_client_info_t *info);
int snd_seq_client_info_get_broadcast_filter(const snd_seq_client_info_t *info);
int snd_seq_client_info_get_error_bounce(const snd_seq_client_info_t *info);
const unsigned char *snd_seq_client_info_get_event_filter(const snd_seq_client_info_t *info);
int snd_seq_client_info_get_num_ports(const snd_seq_client_info_t *info);
int snd_seq_client_info_get_event_lost(const snd_seq_client_info_t *info);

void snd_seq_client_info_set_client(snd_seq_client_info_t *info, int client);
void snd_seq_client_info_set_name(snd_seq_client_info_t *info, const char *name);
void snd_seq_client_info_set_broadcast_filter(snd_seq_client_info_t *info, int val);
void snd_seq_client_info_set_error_bounce(snd_seq_client_info_t *info, int val);
void snd_seq_client_info_set_event_filter(snd_seq_client_info_t *info, unsigned char *filter);

void snd_seq_client_info_event_filter_clear(snd_seq_client_info_t *info);
void snd_seq_client_info_event_filter_add(snd_seq_client_info_t *info, int event_type);
void snd_seq_client_info_event_filter_del(snd_seq_client_info_t *info, int event_type);
int snd_seq_client_info_event_filter_check(snd_seq_client_info_t *info, int event_type);

int snd_seq_get_client_info(snd_seq_t *handle, snd_seq_client_info_t *info);
int snd_seq_get_any_client_info(snd_seq_t *handle, int client, snd_seq_client_info_t *info);
int snd_seq_set_client_info(snd_seq_t *handle, snd_seq_client_info_t *info);
int snd_seq_query_next_client(snd_seq_t *handle, snd_seq_client_info_t *info);





typedef struct _snd_seq_client_pool snd_seq_client_pool_t;

size_t snd_seq_client_pool_sizeof(void);



int snd_seq_client_pool_malloc(snd_seq_client_pool_t **ptr);
void snd_seq_client_pool_free(snd_seq_client_pool_t *ptr);
void snd_seq_client_pool_copy(snd_seq_client_pool_t *dst, const snd_seq_client_pool_t *src);

int snd_seq_client_pool_get_client(const snd_seq_client_pool_t *info);
size_t snd_seq_client_pool_get_output_pool(const snd_seq_client_pool_t *info);
size_t snd_seq_client_pool_get_input_pool(const snd_seq_client_pool_t *info);
size_t snd_seq_client_pool_get_output_room(const snd_seq_client_pool_t *info);
size_t snd_seq_client_pool_get_output_free(const snd_seq_client_pool_t *info);
size_t snd_seq_client_pool_get_input_free(const snd_seq_client_pool_t *info);
void snd_seq_client_pool_set_output_pool(snd_seq_client_pool_t *info, size_t size);
void snd_seq_client_pool_set_input_pool(snd_seq_client_pool_t *info, size_t size);
void snd_seq_client_pool_set_output_room(snd_seq_client_pool_t *info, size_t size);

int snd_seq_get_client_pool(snd_seq_t *handle, snd_seq_client_pool_t *info);
int snd_seq_set_client_pool(snd_seq_t *handle, snd_seq_client_pool_t *info);
# 205 "/usr/include/alsa/seq.h" 3
typedef struct _snd_seq_port_info snd_seq_port_info_t;
# 261 "/usr/include/alsa/seq.h" 3
size_t snd_seq_port_info_sizeof(void);



int snd_seq_port_info_malloc(snd_seq_port_info_t **ptr);
void snd_seq_port_info_free(snd_seq_port_info_t *ptr);
void snd_seq_port_info_copy(snd_seq_port_info_t *dst, const snd_seq_port_info_t *src);

int snd_seq_port_info_get_client(const snd_seq_port_info_t *info);
int snd_seq_port_info_get_port(const snd_seq_port_info_t *info);
const snd_seq_addr_t *snd_seq_port_info_get_addr(const snd_seq_port_info_t *info);
const char *snd_seq_port_info_get_name(const snd_seq_port_info_t *info);
unsigned int snd_seq_port_info_get_capability(const snd_seq_port_info_t *info);
unsigned int snd_seq_port_info_get_type(const snd_seq_port_info_t *info);
int snd_seq_port_info_get_midi_channels(const snd_seq_port_info_t *info);
int snd_seq_port_info_get_midi_voices(const snd_seq_port_info_t *info);
int snd_seq_port_info_get_synth_voices(const snd_seq_port_info_t *info);
int snd_seq_port_info_get_read_use(const snd_seq_port_info_t *info);
int snd_seq_port_info_get_write_use(const snd_seq_port_info_t *info);
int snd_seq_port_info_get_port_specified(const snd_seq_port_info_t *info);
int snd_seq_port_info_get_timestamping(const snd_seq_port_info_t *info);
int snd_seq_port_info_get_timestamp_real(const snd_seq_port_info_t *info);
int snd_seq_port_info_get_timestamp_queue(const snd_seq_port_info_t *info);

void snd_seq_port_info_set_client(snd_seq_port_info_t *info, int client);
void snd_seq_port_info_set_port(snd_seq_port_info_t *info, int port);
void snd_seq_port_info_set_addr(snd_seq_port_info_t *info, const snd_seq_addr_t *addr);
void snd_seq_port_info_set_name(snd_seq_port_info_t *info, const char *name);
void snd_seq_port_info_set_capability(snd_seq_port_info_t *info, unsigned int capability);
void snd_seq_port_info_set_type(snd_seq_port_info_t *info, unsigned int type);
void snd_seq_port_info_set_midi_channels(snd_seq_port_info_t *info, int channels);
void snd_seq_port_info_set_midi_voices(snd_seq_port_info_t *info, int voices);
void snd_seq_port_info_set_synth_voices(snd_seq_port_info_t *info, int voices);
void snd_seq_port_info_set_port_specified(snd_seq_port_info_t *info, int val);
void snd_seq_port_info_set_timestamping(snd_seq_port_info_t *info, int enable);
void snd_seq_port_info_set_timestamp_real(snd_seq_port_info_t *info, int realtime);
void snd_seq_port_info_set_timestamp_queue(snd_seq_port_info_t *info, int queue);

int snd_seq_create_port(snd_seq_t *handle, snd_seq_port_info_t *info);
int snd_seq_delete_port(snd_seq_t *handle, int port);
int snd_seq_get_port_info(snd_seq_t *handle, int port, snd_seq_port_info_t *info);
int snd_seq_get_any_port_info(snd_seq_t *handle, int client, int port, snd_seq_port_info_t *info);
int snd_seq_set_port_info(snd_seq_t *handle, int port, snd_seq_port_info_t *info);
int snd_seq_query_next_port(snd_seq_t *handle, snd_seq_port_info_t *info);
# 317 "/usr/include/alsa/seq.h" 3
typedef struct _snd_seq_port_subscribe snd_seq_port_subscribe_t;

size_t snd_seq_port_subscribe_sizeof(void);



int snd_seq_port_subscribe_malloc(snd_seq_port_subscribe_t **ptr);
void snd_seq_port_subscribe_free(snd_seq_port_subscribe_t *ptr);
void snd_seq_port_subscribe_copy(snd_seq_port_subscribe_t *dst, const snd_seq_port_subscribe_t *src);

const snd_seq_addr_t *snd_seq_port_subscribe_get_sender(const snd_seq_port_subscribe_t *info);
const snd_seq_addr_t *snd_seq_port_subscribe_get_dest(const snd_seq_port_subscribe_t *info);
int snd_seq_port_subscribe_get_queue(const snd_seq_port_subscribe_t *info);
int snd_seq_port_subscribe_get_exclusive(const snd_seq_port_subscribe_t *info);
int snd_seq_port_subscribe_get_time_update(const snd_seq_port_subscribe_t *info);
int snd_seq_port_subscribe_get_time_real(const snd_seq_port_subscribe_t *info);

void snd_seq_port_subscribe_set_sender(snd_seq_port_subscribe_t *info, const snd_seq_addr_t *addr);
void snd_seq_port_subscribe_set_dest(snd_seq_port_subscribe_t *info, const snd_seq_addr_t *addr);
void snd_seq_port_subscribe_set_queue(snd_seq_port_subscribe_t *info, int q);
void snd_seq_port_subscribe_set_exclusive(snd_seq_port_subscribe_t *info, int val);
void snd_seq_port_subscribe_set_time_update(snd_seq_port_subscribe_t *info, int val);
void snd_seq_port_subscribe_set_time_real(snd_seq_port_subscribe_t *info, int val);

int snd_seq_get_port_subscription(snd_seq_t *handle, snd_seq_port_subscribe_t *sub);
int snd_seq_subscribe_port(snd_seq_t *handle, snd_seq_port_subscribe_t *sub);
int snd_seq_unsubscribe_port(snd_seq_t *handle, snd_seq_port_subscribe_t *sub);





typedef struct _snd_seq_query_subscribe snd_seq_query_subscribe_t;


typedef enum {
 SND_SEQ_QUERY_SUBS_READ,
 SND_SEQ_QUERY_SUBS_WRITE
} snd_seq_query_subs_type_t;

size_t snd_seq_query_subscribe_sizeof(void);



int snd_seq_query_subscribe_malloc(snd_seq_query_subscribe_t **ptr);
void snd_seq_query_subscribe_free(snd_seq_query_subscribe_t *ptr);
void snd_seq_query_subscribe_copy(snd_seq_query_subscribe_t *dst, const snd_seq_query_subscribe_t *src);

int snd_seq_query_subscribe_get_client(const snd_seq_query_subscribe_t *info);
int snd_seq_query_subscribe_get_port(const snd_seq_query_subscribe_t *info);
const snd_seq_addr_t *snd_seq_query_subscribe_get_root(const snd_seq_query_subscribe_t *info);
snd_seq_query_subs_type_t snd_seq_query_subscribe_get_type(const snd_seq_query_subscribe_t *info);
int snd_seq_query_subscribe_get_index(const snd_seq_query_subscribe_t *info);
int snd_seq_query_subscribe_get_num_subs(const snd_seq_query_subscribe_t *info);
const snd_seq_addr_t *snd_seq_query_subscribe_get_addr(const snd_seq_query_subscribe_t *info);
int snd_seq_query_subscribe_get_queue(const snd_seq_query_subscribe_t *info);
int snd_seq_query_subscribe_get_exclusive(const snd_seq_query_subscribe_t *info);
int snd_seq_query_subscribe_get_time_update(const snd_seq_query_subscribe_t *info);
int snd_seq_query_subscribe_get_time_real(const snd_seq_query_subscribe_t *info);

void snd_seq_query_subscribe_set_client(snd_seq_query_subscribe_t *info, int client);
void snd_seq_query_subscribe_set_port(snd_seq_query_subscribe_t *info, int port);
void snd_seq_query_subscribe_set_root(snd_seq_query_subscribe_t *info, const snd_seq_addr_t *addr);
void snd_seq_query_subscribe_set_type(snd_seq_query_subscribe_t *info, snd_seq_query_subs_type_t type);
void snd_seq_query_subscribe_set_index(snd_seq_query_subscribe_t *info, int _index);

int snd_seq_query_port_subscribers(snd_seq_t *seq, snd_seq_query_subscribe_t * subs);
# 396 "/usr/include/alsa/seq.h" 3
typedef struct _snd_seq_queue_info snd_seq_queue_info_t;

typedef struct _snd_seq_queue_status snd_seq_queue_status_t;

typedef struct _snd_seq_queue_tempo snd_seq_queue_tempo_t;

typedef struct _snd_seq_queue_timer snd_seq_queue_timer_t;




size_t snd_seq_queue_info_sizeof(void);



int snd_seq_queue_info_malloc(snd_seq_queue_info_t **ptr);
void snd_seq_queue_info_free(snd_seq_queue_info_t *ptr);
void snd_seq_queue_info_copy(snd_seq_queue_info_t *dst, const snd_seq_queue_info_t *src);

int snd_seq_queue_info_get_queue(const snd_seq_queue_info_t *info);
const char *snd_seq_queue_info_get_name(const snd_seq_queue_info_t *info);
int snd_seq_queue_info_get_owner(const snd_seq_queue_info_t *info);
int snd_seq_queue_info_get_locked(const snd_seq_queue_info_t *info);
unsigned int snd_seq_queue_info_get_flags(const snd_seq_queue_info_t *info);

void snd_seq_queue_info_set_name(snd_seq_queue_info_t *info, const char *name);
void snd_seq_queue_info_set_owner(snd_seq_queue_info_t *info, int owner);
void snd_seq_queue_info_set_locked(snd_seq_queue_info_t *info, int locked);
void snd_seq_queue_info_set_flags(snd_seq_queue_info_t *info, unsigned int flags);

int snd_seq_create_queue(snd_seq_t *seq, snd_seq_queue_info_t *info);
int snd_seq_alloc_named_queue(snd_seq_t *seq, const char *name);
int snd_seq_alloc_queue(snd_seq_t *handle);
int snd_seq_free_queue(snd_seq_t *handle, int q);
int snd_seq_get_queue_info(snd_seq_t *seq, int q, snd_seq_queue_info_t *info);
int snd_seq_set_queue_info(snd_seq_t *seq, int q, snd_seq_queue_info_t *info);
int snd_seq_query_named_queue(snd_seq_t *seq, const char *name);

int snd_seq_get_queue_usage(snd_seq_t *handle, int q);
int snd_seq_set_queue_usage(snd_seq_t *handle, int q, int used);



size_t snd_seq_queue_status_sizeof(void);



int snd_seq_queue_status_malloc(snd_seq_queue_status_t **ptr);
void snd_seq_queue_status_free(snd_seq_queue_status_t *ptr);
void snd_seq_queue_status_copy(snd_seq_queue_status_t *dst, const snd_seq_queue_status_t *src);

int snd_seq_queue_status_get_queue(const snd_seq_queue_status_t *info);
int snd_seq_queue_status_get_events(const snd_seq_queue_status_t *info);
snd_seq_tick_time_t snd_seq_queue_status_get_tick_time(const snd_seq_queue_status_t *info);
const snd_seq_real_time_t *snd_seq_queue_status_get_real_time(const snd_seq_queue_status_t *info);
unsigned int snd_seq_queue_status_get_status(const snd_seq_queue_status_t *info);

int snd_seq_get_queue_status(snd_seq_t *handle, int q, snd_seq_queue_status_t *status);



size_t snd_seq_queue_tempo_sizeof(void);



int snd_seq_queue_tempo_malloc(snd_seq_queue_tempo_t **ptr);
void snd_seq_queue_tempo_free(snd_seq_queue_tempo_t *ptr);
void snd_seq_queue_tempo_copy(snd_seq_queue_tempo_t *dst, const snd_seq_queue_tempo_t *src);

int snd_seq_queue_tempo_get_queue(const snd_seq_queue_tempo_t *info);
unsigned int snd_seq_queue_tempo_get_tempo(const snd_seq_queue_tempo_t *info);
int snd_seq_queue_tempo_get_ppq(const snd_seq_queue_tempo_t *info);
unsigned int snd_seq_queue_tempo_get_skew(const snd_seq_queue_tempo_t *info);
unsigned int snd_seq_queue_tempo_get_skew_base(const snd_seq_queue_tempo_t *info);
void snd_seq_queue_tempo_set_tempo(snd_seq_queue_tempo_t *info, unsigned int tempo);
void snd_seq_queue_tempo_set_ppq(snd_seq_queue_tempo_t *info, int ppq);
void snd_seq_queue_tempo_set_skew(snd_seq_queue_tempo_t *info, unsigned int skew);
void snd_seq_queue_tempo_set_skew_base(snd_seq_queue_tempo_t *info, unsigned int base);

int snd_seq_get_queue_tempo(snd_seq_t *handle, int q, snd_seq_queue_tempo_t *tempo);
int snd_seq_set_queue_tempo(snd_seq_t *handle, int q, snd_seq_queue_tempo_t *tempo);





typedef enum {
 SND_SEQ_TIMER_ALSA = 0,
 SND_SEQ_TIMER_MIDI_CLOCK = 1,
 SND_SEQ_TIMER_MIDI_TICK = 2
} snd_seq_queue_timer_type_t;

size_t snd_seq_queue_timer_sizeof(void);



int snd_seq_queue_timer_malloc(snd_seq_queue_timer_t **ptr);
void snd_seq_queue_timer_free(snd_seq_queue_timer_t *ptr);
void snd_seq_queue_timer_copy(snd_seq_queue_timer_t *dst, const snd_seq_queue_timer_t *src);

int snd_seq_queue_timer_get_queue(const snd_seq_queue_timer_t *info);
snd_seq_queue_timer_type_t snd_seq_queue_timer_get_type(const snd_seq_queue_timer_t *info);
const snd_timer_id_t *snd_seq_queue_timer_get_id(const snd_seq_queue_timer_t *info);
unsigned int snd_seq_queue_timer_get_resolution(const snd_seq_queue_timer_t *info);

void snd_seq_queue_timer_set_type(snd_seq_queue_timer_t *info, snd_seq_queue_timer_type_t type);
void snd_seq_queue_timer_set_id(snd_seq_queue_timer_t *info, const snd_timer_id_t *id);
void snd_seq_queue_timer_set_resolution(snd_seq_queue_timer_t *info, unsigned int resolution);

int snd_seq_get_queue_timer(snd_seq_t *handle, int q, snd_seq_queue_timer_t *timer);
int snd_seq_set_queue_timer(snd_seq_t *handle, int q, snd_seq_queue_timer_t *timer);
# 517 "/usr/include/alsa/seq.h" 3
int snd_seq_free_event(snd_seq_event_t *ev);
ssize_t snd_seq_event_length(snd_seq_event_t *ev);
int snd_seq_event_output(snd_seq_t *handle, snd_seq_event_t *ev);
int snd_seq_event_output_buffer(snd_seq_t *handle, snd_seq_event_t *ev);
int snd_seq_event_output_direct(snd_seq_t *handle, snd_seq_event_t *ev);
int snd_seq_event_input(snd_seq_t *handle, snd_seq_event_t **ev);
int snd_seq_event_input_pending(snd_seq_t *seq, int fetch_sequencer);
int snd_seq_drain_output(snd_seq_t *handle);
int snd_seq_event_output_pending(snd_seq_t *seq);
int snd_seq_extract_output(snd_seq_t *handle, snd_seq_event_t **ev);
int snd_seq_drop_output(snd_seq_t *handle);
int snd_seq_drop_output_buffer(snd_seq_t *handle);
int snd_seq_drop_input(snd_seq_t *handle);
int snd_seq_drop_input_buffer(snd_seq_t *handle);


typedef struct _snd_seq_remove_events snd_seq_remove_events_t;
# 547 "/usr/include/alsa/seq.h" 3
size_t snd_seq_remove_events_sizeof(void);



int snd_seq_remove_events_malloc(snd_seq_remove_events_t **ptr);
void snd_seq_remove_events_free(snd_seq_remove_events_t *ptr);
void snd_seq_remove_events_copy(snd_seq_remove_events_t *dst, const snd_seq_remove_events_t *src);

unsigned int snd_seq_remove_events_get_condition(const snd_seq_remove_events_t *info);
int snd_seq_remove_events_get_queue(const snd_seq_remove_events_t *info);
const snd_seq_timestamp_t *snd_seq_remove_events_get_time(const snd_seq_remove_events_t *info);
const snd_seq_addr_t *snd_seq_remove_events_get_dest(const snd_seq_remove_events_t *info);
int snd_seq_remove_events_get_channel(const snd_seq_remove_events_t *info);
int snd_seq_remove_events_get_event_type(const snd_seq_remove_events_t *info);
int snd_seq_remove_events_get_tag(const snd_seq_remove_events_t *info);

void snd_seq_remove_events_set_condition(snd_seq_remove_events_t *info, unsigned int flags);
void snd_seq_remove_events_set_queue(snd_seq_remove_events_t *info, int queue);
void snd_seq_remove_events_set_time(snd_seq_remove_events_t *info, const snd_seq_timestamp_t *time);
void snd_seq_remove_events_set_dest(snd_seq_remove_events_t *info, const snd_seq_addr_t *addr);
void snd_seq_remove_events_set_channel(snd_seq_remove_events_t *info, int channel);
void snd_seq_remove_events_set_event_type(snd_seq_remove_events_t *info, int type);
void snd_seq_remove_events_set_tag(snd_seq_remove_events_t *info, int tag);

int snd_seq_remove_events(snd_seq_t *handle, snd_seq_remove_events_t *info);
# 582 "/usr/include/alsa/seq.h" 3
void snd_seq_set_bit(int nr, void *array);
void snd_seq_unset_bit(int nr, void *array);
int snd_seq_change_bit(int nr, void *array);
int snd_seq_get_bit(int nr, void *array);
# 598 "/usr/include/alsa/seq.h" 3
enum {
 SND_SEQ_EVFLG_RESULT,
 SND_SEQ_EVFLG_NOTE,
 SND_SEQ_EVFLG_CONTROL,
 SND_SEQ_EVFLG_QUEUE,
 SND_SEQ_EVFLG_SYSTEM,
 SND_SEQ_EVFLG_MESSAGE,
 SND_SEQ_EVFLG_CONNECTION,
 SND_SEQ_EVFLG_SAMPLE,
 SND_SEQ_EVFLG_USERS,
 SND_SEQ_EVFLG_INSTR,
 SND_SEQ_EVFLG_QUOTE,
 SND_SEQ_EVFLG_NONE,
 SND_SEQ_EVFLG_RAW,
 SND_SEQ_EVFLG_FIXED,
 SND_SEQ_EVFLG_VARIABLE,
 SND_SEQ_EVFLG_VARUSR
};

enum {
 SND_SEQ_EVFLG_NOTE_ONEARG,
 SND_SEQ_EVFLG_NOTE_TWOARG
};

enum {
 SND_SEQ_EVFLG_QUEUE_NOARG,
 SND_SEQ_EVFLG_QUEUE_TICK,
 SND_SEQ_EVFLG_QUEUE_TIME,
 SND_SEQ_EVFLG_QUEUE_VALUE
};






extern const unsigned int snd_seq_event_types[];
# 57 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/seqmid.h" 1 3
# 288 "/usr/include/alsa/seqmid.h" 3
int snd_seq_control_queue(snd_seq_t *seq, int q, int type, int value, snd_seq_event_t *ev);
# 328 "/usr/include/alsa/seqmid.h" 3
int snd_seq_create_simple_port(snd_seq_t *seq, const char *name,
          unsigned int caps, unsigned int type);

int snd_seq_delete_simple_port(snd_seq_t *seq, int port);




int snd_seq_connect_from(snd_seq_t *seq, int my_port, int src_client, int src_port);
int snd_seq_connect_to(snd_seq_t *seq, int my_port, int dest_client, int dest_port);
int snd_seq_disconnect_from(snd_seq_t *seq, int my_port, int src_client, int src_port);
int snd_seq_disconnect_to(snd_seq_t *seq, int my_port, int dest_client, int dest_port);




int snd_seq_set_client_name(snd_seq_t *seq, const char *name);
int snd_seq_set_client_event_filter(snd_seq_t *seq, int event_type);
int snd_seq_set_client_pool_output(snd_seq_t *seq, size_t size);
int snd_seq_set_client_pool_output_room(snd_seq_t *seq, size_t size);
int snd_seq_set_client_pool_input(snd_seq_t *seq, size_t size);

int snd_seq_sync_output_queue(snd_seq_t *seq);




int snd_seq_parse_address(snd_seq_t *seq, snd_seq_addr_t *addr, const char *str);




int snd_seq_reset_pool_output(snd_seq_t *seq);
int snd_seq_reset_pool_input(snd_seq_t *seq);
# 58 "/usr/include/alsa/asoundlib.h" 2 3
# 1 "/usr/include/alsa/seq_midi_event.h" 1 3
# 43 "/usr/include/alsa/seq_midi_event.h" 3
typedef struct snd_midi_event snd_midi_event_t;

int snd_midi_event_new(size_t bufsize, snd_midi_event_t **rdev);
int snd_midi_event_resize_buffer(snd_midi_event_t *dev, size_t bufsize);
void snd_midi_event_free(snd_midi_event_t *dev);
void snd_midi_event_init(snd_midi_event_t *dev);
void snd_midi_event_reset_encode(snd_midi_event_t *dev);
void snd_midi_event_reset_decode(snd_midi_event_t *dev);
void snd_midi_event_no_status(snd_midi_event_t *dev, int on);

long snd_midi_event_encode(snd_midi_event_t *dev, const unsigned char *buf, long count, snd_seq_event_t *ev);
int snd_midi_event_encode_byte(snd_midi_event_t *dev, int c, snd_seq_event_t *ev);

long snd_midi_event_decode(snd_midi_event_t *dev, unsigned char *buf, long count, const snd_seq_event_t *ev);
# 59 "/usr/include/alsa/asoundlib.h" 2 3
# 60 "sound.h" 2
# 74 "sound.h"
extern sliderContainer_t soundSlider;
extern snd_mixer_t *alsaMixerHandle;
# 86 "sound.h"
extern void sound_buildSlider(void);
# 95 "sound.h"
extern void sound_setVolume(long value);






extern int32_t sound_init(void);






extern void sound_freeSlider(void);
# 140 "exStbDemo.c" 2
# 1 "pvr.h" 1
# 71 "pvr.h"
extern menuContainer_t PvrMenu;
extern sliderContainer_t pvrSlider;
# 89 "pvr.h"
extern void pvr_getPvrInformation(char* filename, char* pChannelName, char* pRecordDate, char* pRecordTime, int32_t skipInfoFile);
# 98 "pvr.h"
extern void pvr_stopRecording(uint32_t which);
# 108 "pvr.h"
extern void pvr_startPlayback(uint32_t which, int32_t fromStart);
# 117 "pvr.h"
extern void pvr_stopPlayback(uint32_t which);






extern void pvr_buildPvrMenu(void);






extern void pvr_buildSlider(void);
# 142 "pvr.h"
extern void pvr_setPosition(int32_t which, long value);
# 151 "pvr.h"
extern void pvr_setDirection(trickModeDirection_t direction);
# 160 "pvr.h"
extern void pvr_setSpeed(trickModeSpeed_t speed);






extern void pvr_trickModeStart(void);
# 176 "pvr.h"
extern void pvr_trickModeStop(int32_t startPvr);
# 185 "pvr.h"
extern void pvr_trickModeSetup(int32_t restart);






extern void pvr_pause(void);






extern void pvr_recordNow(void);
# 208 "pvr.h"
extern void pvr_forwards(int32_t seconds);
# 217 "pvr.h"
extern void pvr_backwards(int32_t seconds);
# 227 "pvr.h"
extern void pvr_getFilename(uint32_t which, uint32_t fileNumber);






extern void pvr_freeMenu(void);
# 141 "exStbDemo.c" 2
# 1 "iprtp.h" 1
# 76 "iprtp.h"
extern menuContainer_t ipMenu;







extern void ip_parseInfo(void);







extern void ip_buildMenu(void);
# 102 "iprtp.h"
extern void ip_setStreamNumber(uint32_t which, uint32_t streamNumber);
# 111 "iprtp.h"
extern void ip_startVideo(uint32_t which);
# 120 "iprtp.h"
extern void ip_stopVideo(uint32_t which);
# 137 "iprtp.h"
extern void ip_freeMenu(void);
# 146 "iprtp.h"
extern void ip_switchAudioPid( uint32_t pidIndex);
# 142 "exStbDemo.c" 2
# 1 "prog_info.h" 1
# 80 "prog_info.h"
extern void initProgramInfo(programInfo_t * pProgram);
# 89 "prog_info.h"
extern void getProgramInfo(programInfo_t * pProgram);
# 98 "prog_info.h"
extern void getTimeInfo(timeInfo_t * pTime);
# 143 "exStbDemo.c" 2
# 1 "media.h" 1
# 85 "media.h"
extern menuContainer_t MediaMenu;
extern sliderContainer_t mediaSlider;
# 97 "media.h"
extern void media_startPlayback(void);






extern void media_stopPlayback(void);






extern void media_buildMediaMenu(void);






extern void media_buildSlider(void);
# 127 "media.h"
extern void media_setPosition(long value);
# 136 "media.h"
extern void media_getFilename(int32_t fileNumber);






extern void media_pause(void);
# 152 "media.h"
extern void media_setTrickModeDirection(trickModeDirection_t direction);
# 161 "media.h"
extern void media_setTrickModeSpeed(trickModeSpeed_t speed);






extern void media_increaseTrickModeSpeed(void);






extern void media_decreaseTrickModeSpeed(void);
# 184 "media.h"
extern void media_setTrickModeActive(int active);
# 193 "media.h"
extern void media_seek(off_t offset);






extern void media_freeMenu(void);
# 210 "media.h"
extern void media_switchAudioPid( uint32_t pid);
# 144 "exStbDemo.c" 2
# 1 "monitoring.h" 1
# 69 "monitoring.h"
typedef enum {
    monitoring_video,
    monitoring_audio,
    monitoring_cpu2,
    monitoring_cpu1,
    monitoring_streaming,
    monitoring_decode,
    monitoring_guard
}monitoring_data;





extern menuContainer_t MonitoringMenu;
# 94 "monitoring.h"
extern void monitoring_buildMonitoringMenu(void);






extern void monitoring_freeMenu(void);






extern void monitoring_init(void);






extern void monitoring_term(void);
# 125 "monitoring.h"
extern uint32_t monitoring_getTrackLevel(monitoring_data which, _Bool high);
# 137 "monitoring.h"
extern void monitoring_drawUsage(monitoring_data which, uint32_t offset, uint32_t low, uint32_t high);






extern void monitoring_drawHistogram(void);






extern void monitoring_drawEvents(void);
# 145 "exStbDemo.c" 2
# 1 "output.h" 1
# 75 "output.h"
extern menuContainer_t OutputMenu;
# 86 "output.h"
extern void output_buildOutputMenu(void);






extern void output_freeMenu(void);
# 146 "exStbDemo.c" 2
# 182 "exStbDemo.c"
typedef enum __commandCode
{
    commandMenu,
    commandUp,
    commandDown,
    commandLeft,
    commandRight,
    commandChannelUp,
    commandChannelDown,
    commandVolumeUp,
    commandVolumeDown,
    commandMute,
    commandOK,
    commandInfo,
    commandDigit0,
    commandDigit1,
    commandDigit2,
    commandDigit3,
    commandDigit4,
    commandDigit5,
    commandDigit6,
    commandDigit7,
    commandDigit8,
    commandDigit9,
    commandShutdown,
    commandChangePipPosition,
    commandPause,
    commandRed,
    commandYellow,
    commandBlue,
    commandNone,
} commandCode;
# 224 "exStbDemo.c"



static char *commandName[] = {
    "Menu",
    "Up",
    "Down",
    "Left",
    "Right",
    "ChannelUp",
    "ChannelDown",
    "VolumeUp",
    "VolumeDown",
    "Mute",
    "OK",
    "Info",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "Shutdown",
    "ChangePipPosition",
    "Pause",
    "Red",
    "Yellow",
    "Blue",
    "None",
};

static struct timeval commandTime[2];
static volatile _Bool keepCommandLoopAlive = 1;
static IDirectFBEventBuffer *eventBuffer = ((void *)0);
static volatile _Bool mainLoopActive = 1;


static tmInstance_t gAIO;
# 276 "exStbDemo.c"
void exStbDemo_threadRename(char* name)
{
    char newName[16];

    (void)snprintf(newName, 15, "#%s", name);
    (void)prctl(15, newName, 0,0,0);
}

int32_t exStbDemo_fileExists(char* filename)
{
    int32_t exists = 0;
    struct stat buffer;
    int32_t status;

    if(strlen(filename) > 0)
    {
        status = stat(filename, &buffer);
        if (status < 0)
        {
            do{}while(0);
        }
        else
        {

            exists = !(((((buffer.st_mode)) & 0170000) == (0040000)));
        }
    }
    return exists;
}

static int32_t readLine(int32_t file, char* buffer)
{
    if (file)
    {
        int32_t lineIndex = 0;
        while (1)
        {
            char c;

            if (read(file, &c, 1) < 1)
            {
                return -1;
            }

            if (c == '\n')
            {
               buffer[lineIndex] = '\0';
               return 0;
            }
            else
            {
                buffer[lineIndex] = c;
                lineIndex++;
            }
        }
    }
    return -1;
}

static commandCode getCommand(IDirectFBEventBuffer *commandEventBuffer, int32_t flush)
{
    DFBEvent event;
    commandCode command = commandNone;

    if (appControlInfo.commandInfo.inputFile)
    {
        char buffer[128];

        do{}while(0);

        if (-1==readLine(appControlInfo.commandInfo.inputFile, buffer))
        {

            if (appControlInfo.commandInfo.loop)
            {

                (void)lseek(appControlInfo.commandInfo.inputFile, 0L, 0);
                (void)readLine(appControlInfo.commandInfo.inputFile, buffer);
            }
            else
            {

                appControlInfo.commandInfo.inputFile = 0;
            }
        }


        if (appControlInfo.commandInfo.inputFile)
        {
            long timeSinceLastCommand;
            long delay;
            char commandText[128];
            static int32_t randomNumber = 0;
            static int32_t divisor = 1;
            static struct timeval loggedCommandTime;
            (void)sscanf(buffer, "%ld %s", &delay, commandText);
            do{}while(0);


            for(command=commandMenu; command<commandNone; command++)
            {
                if (!strcmp(commandText, commandName[command]))
                {

                    break;
                }
            }

            if (!strcmp(commandText, "RandomOn"))
            {

                randomNumber = 1;
                divisor = delay;
            }

            if (!strcmp(commandText, "RandomOff"))
            {
                randomNumber = 0;
            }

            (void)printf("Command : %s delay %ld ms", commandText, delay);
            if (randomNumber)
            {

                delay /= (1 + (rand() % divisor));
                (void)printf(" -> randomised delay %ld ms\n", delay);
            }
            (void)printf("\n");

            do
            {

                (void)usleep(10000);
                (void)gettimeofday(&loggedCommandTime, ((void *)0));
                timeSinceLastCommand = (((long)(loggedCommandTime.tv_sec - commandTime[(0)].tv_sec) * 1000) +
                                        ((loggedCommandTime.tv_usec - commandTime[(0)].tv_usec)/1000));
            }while(timeSinceLastCommand < delay);
        }
    }
    else
    {
        int32_t gotEvent = 0;

        if (flush)
        {
            int32_t gotCommand;
            do
            {
                gotCommand = 0;
                commandEventBuffer->WaitForEventWithTimeout(commandEventBuffer, 0, 1);


                if (commandEventBuffer->HasEvent(commandEventBuffer) == DFB_OK)
                {
                    commandEventBuffer->GetEvent(commandEventBuffer, &event);
                    do{}while(0);
                    gotCommand = 1;
                }
            }while(gotCommand);

        }
        do{}while(0);

  if (commandEventBuffer != ((void *)0))
  {
         commandEventBuffer->WaitForEventWithTimeout(commandEventBuffer, 1, 0);
  }

  if (commandEventBuffer != ((void *)0))
  {

        if (commandEventBuffer->HasEvent(commandEventBuffer) == DFB_OK)
        {
            commandEventBuffer->GetEvent(commandEventBuffer, &event);

            if ((event.input.type == DIET_BUTTONPRESS) ||
                (event.input.type == DIET_KEYPRESS))
            {
                int32_t continueFlushing;
                gotEvent = 1;

                do
                {
                    DFBEvent tempEvent;
                    continueFlushing = 0;
                    commandEventBuffer->WaitForEventWithTimeout(commandEventBuffer, 0, 1);
                    if (commandEventBuffer->HasEvent(commandEventBuffer) == DFB_OK)
                    {
                        commandEventBuffer->GetEvent(commandEventBuffer, &tempEvent);
                        continueFlushing = 1;
                    }
                }while(continueFlushing);
            }
        }
  }

        if (gotEvent)
        {
            if (event.input.type == DIET_KEYPRESS)
            {
                if (appControlInfo.soundInfo.keyBeep)
                {
                    (void)tmdlAudioIO_AudioBeep(gAIO, 12 );
                }

                if (event.input.flags&DIEF_KEYID)
                {

                    switch(event.input.key_id)
                    {
                        case(DIKI_KP_3) :
                            command = commandMenu;
                            break;
                        case(DIKI_KP_0) :
                            command = commandLeft;
                            break;
                        case(DIKI_KP_6) :
                            command = commandRight;
                            break;
                        case(DIKI_KP_5) :
                            command = commandDown;
                            break;
                        case(DIKI_KP_4) :
                            command = commandUp;
                            break;
                        case(DIKI_KP_9) :
                            command = commandShutdown;
                            break;
                        default :
                            command = commandNone;
                            break;
                    }
                }

                if (command == commandNone)
                {
                    if (event.input.flags&DIEF_KEYSYMBOL)
                    {

                        switch(event.input.key_symbol)
                        {
                            case(DIKS_0) :
                            case(DIKS_1) :
                            case(DIKS_2) :
                            case(DIKS_3) :
                            case(DIKS_4) :
                            case(DIKS_5) :
                            case(DIKS_6) :
                            case(DIKS_7) :
                            case(DIKS_8) :
                            case(DIKS_9) :
                                command = (commandCode)(commandDigit0 + (event.input.key_symbol - DIKS_0));
                                break;
                            case(DIKS_MENU) :
                                command = commandMenu;
                                break;
                            case(DIKS_CURSOR_LEFT) :
                                command = commandLeft;
                                break;
                            case(DIKS_OK) :
                                command = commandOK;
                                break;
                            case(DIKS_CURSOR_RIGHT) :
                                command = commandRight;
                                break;
                            case(DIKS_CURSOR_DOWN) :
                                command = commandDown;
                                break;
                            case(DIKS_CHANNEL_DOWN) :
                                command = commandChannelDown;
                                break;
                            case(DIKS_CURSOR_UP) :
                                command = commandUp;
                                break;
                            case(DIKS_CHANNEL_UP) :
                                command = commandChannelUp;
                                break;
                            case(DIKS_VOLUME_UP) :
                                command = commandVolumeUp;
                                break;
                            case(DIKS_VOLUME_DOWN) :
                                command = commandVolumeDown;
                                break;
                            case(DIKS_MUTE) :
                                command = commandMute;
                                break;
                            case(DIKS_RED) :
                                command = commandRed;
                                break;
                            case(DIKS_YELLOW) :
                                command = commandYellow;
                                break;
                            case(DIKS_BLUE) :
                                command = commandBlue;
                                break;
                            case((DFBInputDeviceKeySymbol)((DIKT_CUSTOM) | ((1)))):
                                command = commandInfo;
                                break;
                            case(DIKS_POWER) :
                                command = commandShutdown;
                                break;
                            case((DFBInputDeviceKeySymbol)((DIKT_CUSTOM) | ((14)))) :
                                command = commandChangePipPosition;
                                break;
                            case(DIKS_EXIT) :
                                command = commandPause;
                                break;
                            default :
                                command = commandNone;
                                break;
                        }
                    }
                }
            }
        }
    }


    if (appControlInfo.scanActive)
    {
        command = commandNone;
    }

    if (command != commandNone)
    {
        do{}while(0);
    }
    return command;
}

static void powerDown(void)
{
    if (appControlInfo.commandInfo.outputFile)
    {
        (void)close(appControlInfo.commandInfo.outputFile);
    }
    keepCommandLoopAlive = 0;

}

static void digitStart(int32_t value)
{
    if ((appControlInfo.mediaInfo.active) ||
        (appControlInfo.pvrInfo.playbackInfo[screenMain].active) ||
        (appControlInfo.dvbInfo[screenMain].active))
    {
        appControlInfo.digitInfo.active = 1;
        appControlInfo.digitInfo.value = (uint32_t)value;
        appControlInfo.digitInfo.numDigits = 1;
        appControlInfo.digitInfo.countDown = (4);
        menuInfra_updateOSD();
    }
}

static void digitAdd(int32_t value)
{
    if ((appControlInfo.mediaInfo.active) ||
        (appControlInfo.pvrInfo.playbackInfo[screenMain].active) ||
        (appControlInfo.dvbInfo[screenMain].active))
    {
        if (value >= 0)
        {
            appControlInfo.digitInfo.value = (appControlInfo.digitInfo.value*10 + (uint32_t)value);
        }
        appControlInfo.digitInfo.numDigits++;
        menuInfra_updateOSD();

        if ((value < 0) || (appControlInfo.digitInfo.numDigits==3))
        {
            appControlInfo.digitInfo.active = 0;
            appControlInfo.digitInfo.numDigits = 0;

            if ( appControlInfo.pvrInfo.playbackInfo[screenMain].active )
            {

                if (appControlInfo.digitInfo.value <= 100)
                {
                    pvr_setPosition(screenMain, (long)appControlInfo.digitInfo.value);
                }
            }
            else
            {
                menuApp_channelChange(appControlInfo.digitInfo.value);
            }
        }

        menuInfra_updateOSD();
    }
}

static void checkEndOfPvrStream(void)
{
    screenOutput_t which;

    for ( which = screenMain; which < screenOutputs; which++ )
    {
        if ( (appControlInfo.pvrInfo.playbackInfo[which].active) &&
             (!appControlInfo.pvrInfo.playbackInfo[which].paused) &&
             (appControlInfo.pvrInfo.playbackInfo[which].endOfFile) )
        {

            if ( appControlInfo.pvrInfo.playbackMode == playback_sequential )
            {
                uint32_t nextFile;
                pvr_stopPlayback((uint32_t)which);
                nextFile = appControlInfo.pvrInfo.playbackInfo[which].fileNumber + 1;
                if ( nextFile == appControlInfo.pvrInfo.maxFile )
                {
                    nextFile = 0;
                }
                pvr_getFilename((uint32_t)which, nextFile);
                pvr_startPlayback((uint32_t)which, 1);
            }
            else
            {
                if ( appControlInfo.pvrInfo.playbackMode == playback_looped )
                {
                    pvr_setPosition((int32_t)which, 0);
                }
            }
        }
    }
}

static void checkEndOfIPStream(void)
{
    if ( (appControlInfo.ipInfo[screenMain].active) &&
         (appControlInfo.ipInfo[screenMain].endOfFile))
    {
        if ( appControlInfo.ipInfo[screenMain].playbackMode == playback_sequential )
        {
            uint32_t nextFile;
            ip_stopVideo(screenMain);
            nextFile = appControlInfo.ipInfo[screenMain].streamNumber + 1;
            if ( nextFile == appControlInfo.ipInfo[screenMain].maxStreams )
            {

                nextFile = 1;
            }
            ip_setStreamNumber(screenMain, nextFile);
            ip_startVideo(screenMain);
        }
        else
        {
            if ( appControlInfo.ipInfo[screenMain].playbackMode == playback_looped )
            {
                ip_stopVideo(screenMain);
                ip_startVideo(screenMain);
            }
        }
    }
}

static void checkEndOfMediaStream(void)
{



    if( appControlInfo.mediaInfo.endOfStreamReported )
    {
        if ( appControlInfo.mediaInfo.playbackMode == playback_sequential )
        {
            int32_t nextFile;
            media_stopPlayback();
            nextFile = appControlInfo.mediaInfo.currentFile + 1;
            if ( nextFile == appControlInfo.mediaInfo.maxFile )
            {
                nextFile = 0;
            }
            media_getFilename(nextFile);
            media_startPlayback();
        }
        else
        {
            if ( appControlInfo.mediaInfo.playbackMode == playback_looped )
            {
                media_stopPlayback();
                media_startPlayback();
            }
        }
    }
}

static void commandLoop( IDirectFBEventBuffer *commandEventBuffer )
{
     int32_t flush = 0;
     (void)gettimeofday(&commandTime[(0)], ((void *)0));
     keepCommandLoopAlive = 1;


     appControlInfo.restartGraphics = 0;

     while(keepCommandLoopAlive && !appControlInfo.restartGraphics)
     {
          commandCode command;
          int32_t redraw = 0;


          command = getCommand(commandEventBuffer, flush);


          flush = 0;
          if (command != commandNone)
          {

              flush = 1;


              (void)gettimeofday(&commandTime[(1)], ((void *)0));

              if (appControlInfo.commandInfo.outputFile)
              {

                  char buffer[1024];
                  long timeSinceLastCommand;

                  timeSinceLastCommand = (((long)(commandTime[(1)].tv_sec - commandTime[(0)].tv_sec) * 1000) +
                                          ((commandTime[(1)].tv_usec - commandTime[(0)].tv_usec)/1000));


                  (void)sprintf(buffer, "%ld %s", timeSinceLastCommand, commandName[command]);

                  if (pCurrentMenu)
                  {
                      (void)strcat(buffer, " ( ");
                      if (pCurrentMenu->prev)
                      {
                          (void)strcat(buffer, pCurrentMenu->prev->menuEntry[pCurrentMenu->prev->highlight].info);
                      }
                      else
                      {
                          (void)strcat(buffer, "Top Level ");
                      }
                      (void)strcat(buffer, " ");
                      (void)strcat(buffer, pCurrentMenu->menuEntry[pCurrentMenu->highlight].info);
                      (void)strcat(buffer, " )\n");
                  }
                  else
                  {
                      (void)strcat(buffer, " ( No Menu Displayed )\n");
                  }
                  do{}while(0);

                  (void)write(appControlInfo.commandInfo.outputFile, buffer, strlen(buffer));
              }


              commandTime[(0)] = commandTime[(1)];
          }


        if ( appControlInfo.pvrInfo.playbackInfo[screenMain].trickModeInfo.active )
        {
            trickModeDirection_t originalTrickModeDirection;

            originalTrickModeDirection = appControlInfo.pvrInfo.playbackInfo[screenMain].trickModeInfo.direction;

            switch ( command )
            {
            case(commandMenu) :
            case(commandOK) :
                pvr_trickModeStop(1);
                break;
            case(commandLeft) :
                pvr_setDirection(direction_backwards);
                if ( appControlInfo.pvrInfo.playbackInfo[screenMain].trickModeInfo.direction != originalTrickModeDirection )
                {
                    pvr_setSpeed(speed_2);
                    pvr_trickModeSetup(1);
                }
                break;
            case(commandRight) :
                pvr_setDirection(direction_forwards);
                if ( appControlInfo.pvrInfo.playbackInfo[screenMain].trickModeInfo.direction != originalTrickModeDirection )
                {
                    pvr_setSpeed(speed_2);
                    pvr_trickModeSetup(1);
                }
                break;
            case(commandDown) :
                if ( appControlInfo.pvrInfo.playbackInfo[screenMain].trickModeInfo.speed != speed_1_32 )
                {
                    pvr_setSpeed((trickModeSpeed_t)(appControlInfo.pvrInfo.playbackInfo[screenMain].trickModeInfo.speed-1));
                    pvr_trickModeSetup(0);
                }
                break;
            case(commandUp) :
                if ( appControlInfo.pvrInfo.playbackInfo[screenMain].trickModeInfo.speed != speed_32 )
                {
                    pvr_setSpeed((trickModeSpeed_t)(appControlInfo.pvrInfo.playbackInfo[screenMain].trickModeInfo.speed+1));
                    pvr_trickModeSetup(0);
                }
                break;
            case(commandPause) :
                pvr_pause();
                redraw = 1;
                break;
            case(commandShutdown) :
                powerDown();
                break;
            default :
                flush = 0;
                break;
            }
        }
        else if ( appControlInfo.mediaInfo.trickModeInfo.active )
        {
            switch ( command )
            {
            case(commandMenu) :
            case(commandOK) :
                media_setTrickModeActive(0);
                menuInfra_updateOSD();
                break;
            case(commandLeft) :
                break;
            case(commandRight) :
                break;
            case(commandDown) :
                media_decreaseTrickModeSpeed();
                menuInfra_updateOSD();
                break;
            case(commandUp) :
                media_increaseTrickModeSpeed();
                menuInfra_updateOSD();
                break;
            case(commandPause) :
                media_setTrickModeActive(0);
                media_pause();
                redraw = 1;
                break;
            case(commandShutdown) :
                powerDown();
                break;
            default :
                flush = 0;
                break;
            }
        }
        else
          if (pCurrentSlider)
          {
              switch(command)
              {
                  case(commandDigit0) :
                  case(commandDigit1) :
                  case(commandDigit2) :
                  case(commandDigit3) :
                  case(commandDigit4) :
                  case(commandDigit5) :
                  case(commandDigit6) :
                  case(commandDigit7) :
                  case(commandDigit8) :
                  case(commandDigit9) :
                      {
                          int32_t value;
                          value = command-commandDigit0;
                          if (appControlInfo.digitInfo.active)
                          {
                              digitAdd(value);
                          }
                          else
                          {
                              digitStart(value);
                          }
                          command = commandNone;
                      }
                      break;
                  case(commandInfo) :
                      if ( !(pCurrentSlider == &pvrSlider) )
                      {
                          break;
                      }

                  case(commandMenu) :
                  case(commandOK) :
                      menuInfra_sliderClose(pCurrentSlider);
                      break;
                  case(commandLeft) :
                      menuInfra_sliderDecrement(pCurrentSlider);
                      break;
                  case(commandRight) :
                      menuInfra_sliderIncrement(pCurrentSlider);
                      break;
                  case(commandVolumeUp) :
                      if (pCurrentSlider == &soundSlider)
                      {
                          menuInfra_sliderIncrement(pCurrentSlider);
                      }
                      break;
                  case(commandVolumeDown) :
                      if (pCurrentSlider == &soundSlider)
                      {
                          menuInfra_sliderDecrement(pCurrentSlider);
                      }
                      break;
                  case(commandMute) :
                      if (pCurrentMenu == ((void *)0))
                      {

                          if (appControlInfo.soundInfo.muted)
                          {
                              sound_setVolume(appControlInfo.soundInfo.volumeLevel);
                          }
                          else
                          {
                              appControlInfo.soundInfo.muted = 1;
                              sound_setVolume(0);
                          }
                          menuInfra_updateOSD();
                      }
                      break;
                  case(commandShutdown) :
                      powerDown();
                      break;
                  case(commandYellow) :
                      pvr_backwards((15));
                      break;
                 case(commandBlue) :
                      pvr_forwards((15));
                      break;
                  default :
                      flush = 0;
                      break;
              }
          }
          else
          {
              switch(command)
              {
                  case(commandDigit0) :
                  case(commandDigit1) :
                  case(commandDigit2) :
                  case(commandDigit3) :
                  case(commandDigit4) :
                  case(commandDigit5) :
                  case(commandDigit6) :
                  case(commandDigit7) :
                  case(commandDigit8) :
                  case(commandDigit9) :
                      {
                          int32_t value;
                          value = command-commandDigit0;
                          if (appControlInfo.digitInfo.active)
                          {
                              digitAdd(value);
                          }
                          else
                          {
                              digitStart(value);
                          }
                          command = commandNone;
                      }
                      break;
                  case(commandMenu) :
                      if (pCurrentMenu == ((void *)0))
                      {
                          if ( !( appControlInfo.statsInfo.bufferUsage || appControlInfo.statsInfo.dataUsage || appControlInfo.statsInfo.cpuUsage || appControlInfo.statsInfo.displayHistogram || appControlInfo.statsInfo.eventStats ) )
                          {

                              gfx_blendLayer(1, 1, 0, (int32_t)appControlInfo.pictureInfo.osdBlend);
                          }
                          menuApp_displayMain();
                          if ( !( appControlInfo.statsInfo.bufferUsage || appControlInfo.statsInfo.dataUsage || appControlInfo.statsInfo.cpuUsage || appControlInfo.statsInfo.displayHistogram || appControlInfo.statsInfo.eventStats ) )
                          {
                              gfx_blendLayer(0, 1, (3), (int32_t)appControlInfo.pictureInfo.osdBlend);
                          }
                      }
                      else
                      {
                          pCurrentMenu = ((void *)0);
                          if ( !( appControlInfo.statsInfo.bufferUsage || appControlInfo.statsInfo.dataUsage || appControlInfo.statsInfo.cpuUsage || appControlInfo.statsInfo.displayHistogram || appControlInfo.statsInfo.eventStats ) )
                          {

                              gfx_blendLayer(1, 1, (32), (int32_t)appControlInfo.pictureInfo.osdBlend);
                          }
                          menuInfra_updateOSD();
                          if ( !( appControlInfo.statsInfo.bufferUsage || appControlInfo.statsInfo.dataUsage || appControlInfo.statsInfo.cpuUsage || appControlInfo.statsInfo.displayHistogram || appControlInfo.statsInfo.eventStats ) )
                          {
                              gfx_blendLayer(0, 1, 0, (int32_t)appControlInfo.pictureInfo.osdBlend);
                          }
                      }
                      break;
                  case(commandLeft) :
                      if (pCurrentMenu != ((void *)0))
                      {
                          menuApp_displayPrevious();
                      }
                      else
                      {
                          if ( appControlInfo.pvrInfo.playbackInfo[screenMain].active )
                          {

                              if(!appControlInfo.pvrInfo.playbackInfo[screenMain].isH264)
                              {
                                  pvr_setDirection(direction_backwards);
                                  pvr_setSpeed(speed_2);
                                  pvr_trickModeStart();
                                 pCurrentSlider = &pvrSlider;
                              }
                          }
                          else
                          {
                                (void)menuInfra_sliderDisplay((void*)&soundSlider);
                          }
                      }
                      break;
                  case(commandVolumeUp) :
                  case(commandVolumeDown) :
                      if (pCurrentMenu == ((void *)0))
                      {
                          (void)menuInfra_sliderDisplay((void*)&soundSlider);
                      }
                      break;
                  case(commandMute) :

                      if (appControlInfo.soundInfo.muted)
                      {
                          sound_setVolume(appControlInfo.soundInfo.volumeLevel);
                      }
                      else
                      {
                          appControlInfo.soundInfo.muted = 1;
                          sound_setVolume(0);
                      }
                      (void)menuInfra_display((void*)pCurrentMenu);
                      break;
                  case(commandRight) :
                      if (pCurrentMenu == ((void *)0))
                      {
                          if ( appControlInfo.pvrInfo.playbackInfo[screenMain].active )
                          {

                              if(!appControlInfo.pvrInfo.playbackInfo[screenMain].isH264)
                              {
                                  pvr_setDirection(direction_forwards);
                                  pvr_setSpeed(speed_2);
                                  pvr_trickModeStart();
                                  pCurrentSlider = &pvrSlider;
                              }
                          }
                          else if ( appControlInfo.mediaInfo.active )
                          {
                              media_setTrickModeDirection(direction_forwards);
                              media_setTrickModeSpeed(speed_2);
                              media_setTrickModeActive(1);
                              menuInfra_updateOSD();
                          }
                          else
                          {
                              (void)menuInfra_sliderDisplay((void*)&soundSlider);
                          }
                          break;
                      }

                  case(commandOK) :
                      if (pCurrentMenu != ((void *)0))
                      {
                          menuApp_doAction();
                      }
                      break;
                  case(commandDown) :
                      if (pCurrentMenu != ((void *)0))
                      {
                          menuApp_highlightDown();
                          break;
                      }

                  case(commandChannelDown) :
                      if (pCurrentMenu == ((void *)0))
                      {
                          menuApp_channelDown();
                      }
                      else
                      {
                          menuApp_highlightPageDown();
                      }
                      break;
                  case(commandUp) :
                      if (pCurrentMenu != ((void *)0))
                      {
                          menuApp_highlightUp();
                          break;
                      }

                  case(commandChannelUp) :
                      if (pCurrentMenu == ((void *)0))
                      {
                          menuApp_channelUp();
                      }
                else
                {
                    menuApp_highlightPageUp();
                }
                break;
            case(commandInfo) :
                if ( pCurrentMenu == ((void *)0) )
                {
                    appControlInfo.displayCount = 5;
                    if ( appControlInfo.pvrInfo.playbackInfo[screenMain].active )
                    {
                        if ( !appControlInfo.pvrInfo.playbackInfo[screenMain].paused )
                        {
                            pCurrentSlider = &pvrSlider;
                        }
                    }
                    if ( appControlInfo.dvbInfo[screenMain].active )
                    {
                        if ( appControlInfo.programInfo.displayed )
                        {
                            appControlInfo.programInfo.displayed = 0;
                            (void)menuInfra_display((void*)pCurrentMenu);
                        }
                        else
                        {
                            initProgramInfo(&appControlInfo.programInfo);
                            appControlInfo.programInfo.displayed = 1;
                            (void)menuInfra_display((void*)pCurrentMenu);
                            getProgramInfo(&appControlInfo.programInfo);
                            (void)menuInfra_display((void*)pCurrentMenu);
                        }
                    }
                }
                break;
            case(commandPause) :
                if ( appControlInfo.pvrInfo.playbackInfo[screenMain].active )
                {
                    pvr_pause();
                    redraw = 1;
                }
                else
                {
                    if ( appControlInfo.mediaInfo.active )
                    {
                        media_pause();
                        redraw = 1;
                    }
                }
                break;
            case(commandRed) :
                pvr_recordNow();
                break;
            case(commandYellow) :
                pvr_backwards((15));
                break;
            case(commandBlue) :
                pvr_forwards((15));
                      break;
                  case(commandShutdown) :
                      powerDown();
                      break;
                  default :
                      flush = 0;
                      break;
              }
          }

          if ( appControlInfo.pvrInfo.playbackInfo[screenMain].active ||
               appControlInfo.pvrInfo.playbackInfo[screenPip].active )
          {
              checkEndOfPvrStream();
          }

          if ( appControlInfo.ipInfo[screenMain].active)
          {
              checkEndOfIPStream();
          }

          if ( appControlInfo.mediaInfo.active )
          {
              checkEndOfMediaStream();
          }

          if (appControlInfo.digitInfo.active)
          {
              if (command != commandNone)
              {
                  appControlInfo.digitInfo.active = 0;
                  menuInfra_updateOSD();
              }
              else
              {
                  appControlInfo.digitInfo.countDown--;
                  if (appControlInfo.digitInfo.countDown == 0)
                  {
                      digitAdd(-1);
                  }
              }
          }

          if ((appControlInfo.tunerDebug) || (pCurrentSlider))
          {
              redraw = 1;
          }
          else
          {
              if (appControlInfo.timeout)
              {
                  if (appControlInfo.displayCount)
                  {
                      appControlInfo.displayCount--;
                      redraw = 1;
                  }
              }
          }

          if ( appControlInfo.mediaInfo.endOfStreamCountdown )
          {
              appControlInfo.mediaInfo.endOfStreamCountdown--;
              redraw = 1;
          }

          if ( appControlInfo.timeInfo.displayed )
          {
              appControlInfo.timeInfo.displayed--;
              redraw = 1;
          }

          if ( ( appControlInfo.statsInfo.bufferUsage || appControlInfo.statsInfo.dataUsage || appControlInfo.statsInfo.cpuUsage || appControlInfo.statsInfo.displayHistogram || appControlInfo.statsInfo.eventStats ) )
          {
              redraw = 1;
          }

          if (redraw)
          {
              menuInfra_updateOSD();
          }
     }
}

static void checkCommandParams(int32_t argc, char* argv[])
{
    int32_t i;
    int32_t arg_increment;

    (void)memset(appControlInfo.ipInfo[0].streamUrl,0,16);
    (void)strncpy(appControlInfo.ipInfo[0].streamUrl,"\0",16);
    (void)memset(appControlInfo.ipInfo[1].streamUrl,0,16);
    (void)strncpy(appControlInfo.ipInfo[1].streamUrl,"\0",16);
    (void)memset(appControlInfo.ipInfo[0].multicastUrl,0,16);
    (void)strncpy(appControlInfo.ipInfo[0].multicastUrl,"\0",16);
    (void)memset(appControlInfo.ipInfo[1].multicastUrl,0,16);
    (void)strncpy(appControlInfo.ipInfo[1].multicastUrl,"\0",16);

    for(i=1; i<argc; i+=arg_increment)
    {
        arg_increment = 1;
        if (!strcmp(argv[i], "-h"))
        {
            (void)printf("Usage : %s [-h] [-notimeout] [-loop] [-input <input command file>]\n"
                   "        [-output <output command file>]\n"
                   "        [-streamer <channel config file>]\n"
                   "        [-stream_url <ip address>] [-rtsp_port <port number>]\n"
                   "        [-multicast_url <ip address>] [-multicast_port <port number>]\n"
                   "        [-seekfile <filename>]\n"
                   "        [-nopts] [-pts]\n"
                   "        [-noaudio] [-audio]\n"
                   "        [-novideo] [-video]\n"
                   "        [-noiperrors] [-iperrors]\n"
                   "        [-preroll]\n"
                   "        [-pdsupport] [-keybeep]\n", argv[0]);
            exit(0);
        }
        else
        if (!strcmp(argv[i], "-loop"))
        {
            appControlInfo.commandInfo.loop = 1;
        }
        else
        if (!strcmp(argv[i], "-notimeout"))
        {
            appControlInfo.timeout = 0;
        }
        else
        if (!strcmp(argv[i], "-nopts"))
        {
            appControlInfo.ptsLocked = 0;
        }
        else
        if (!strcmp(argv[i], "-pts"))
        {
            appControlInfo.ptsLocked = 1;
        }
        else if (!strcmp(argv[i], "-audio"))
        {
            appControlInfo.enableAudio = 1;
        }
        else if (!strcmp(argv[i], "-noaudio"))
        {
            appControlInfo.enableAudio = 0;
        }
        else if (!strcmp(argv[i], "-video"))
        {
            appControlInfo.enableVideo = 1;
        }
        else if (!strcmp(argv[i], "-novideo"))
        {
            appControlInfo.enableVideo = 0;
        }
        else if (!strcmp(argv[i], "-iperrors"))
        {
            appControlInfo.allowStreamErrors = 1;
        }
        else if (!strcmp(argv[i], "-pdsupport"))
        {
            appControlInfo.mediaInfo.pdSupportEnabled = 1;
        }
        else if (!strcmp(argv[i], "-preroll"))
        {
            appControlInfo.mediaInfo.prerollEnabled = 1;
        }
        else if (!strcmp(argv[i], "-keybeep"))
        {
            appControlInfo.soundInfo.keyBeep = 1;
        }
        else if (!strcmp(argv[i], "-noiperrors"))
        {
            appControlInfo.allowStreamErrors = 0;
        }
        else
        if (!strcmp(argv[i], "-streamer"))
        {
            arg_increment++;
            if ((i+1)<argc)
            {
                (void)strcpy( appControlInfo.channelConfigFile, argv[i+1]);
            }
            else
            {
                (void)printf("Warning : Missing channel config filename\n");
            }
            (void)printf("Info : Using channel config file '%s'\n", appControlInfo.channelConfigFile);
            appControlInfo.streamerInput = 1;
            appControlInfo.tunerInfo[0].status = tunerInactive;
            appControlInfo.tunerInfo[1].status = tunerNotPresent;
        }
        else
        if (!strcmp(argv[i], "-input"))
        {
            arg_increment++;
            if ((i+1)<argc)
            {
                appControlInfo.commandInfo.inputFile = open(argv[i+1], 00);
                if (appControlInfo.commandInfo.inputFile<0)
                {
                    (void)printf("Warning : Unable to open input command file '%s'\n", argv[i+1]);
                    appControlInfo.commandInfo.inputFile = 0;
                }
            }
            else
            {
                (void)printf("Warning : Missing input command filename\n");
            }
        }
        else
        if (!strcmp(argv[i], "-output"))
        {
            arg_increment++;
            if ((i+1)<argc)
            {
                appControlInfo.commandInfo.outputFile = creat(argv[i+1], 0666);
                if (appControlInfo.commandInfo.outputFile<0)
                {
                    (void)printf("Warning : Unable to open output command file '%s'\n", argv[i+1]);
                    appControlInfo.commandInfo.outputFile = 0;
                }
            }
            else
            {
                (void)printf("Warning : Missing output command filename\n");
            }
        }
        else
        if ( !strcmp(argv[i], "-stream_url") )
        {
            arg_increment++;
            if ((i+1)<argc)
            {
                (void)memset(appControlInfo.ipInfo[0].streamUrl,0,16);
                (void)strncpy(appControlInfo.ipInfo[0].streamUrl,argv[i+1],15);
                (void)memset(appControlInfo.ipInfo[1].streamUrl,0,16);
                (void)strncpy(appControlInfo.ipInfo[1].streamUrl,argv[i+1],15);
            }
            else
            {
                (void)printf("Warning : Missing stream url address\n");
            }
        }
        else
        if ( !strcmp(argv[i], "-rtsp_port") )
        {
            arg_increment++;
            if ((i+1)<argc)
            {
                appControlInfo.ipInfo[0].rtspPort = (uint32_t)atoi(argv[i+1]);
                appControlInfo.ipInfo[1].rtspPort = (uint32_t)atoi(argv[i+1]);
                if (appControlInfo.ipInfo[0].rtspPort > 65536)
                {
                    (void)printf("Warning : Invalid port range\n");
                    appControlInfo.ipInfo[0].rtspPort = (uint32_t) ((void *)0);
                    appControlInfo.ipInfo[1].rtspPort = (uint32_t) ((void *)0);
                }

            }
            else
            {
                (void)printf("Warning : Missing RTSP port parameter\n");
            }
        }
        else
        if ( !strcmp(argv[i], "-multicast_url") )
        {
            arg_increment++;
            if ((i+1)<argc)
            {
                (void)memset(appControlInfo.ipInfo[0].multicastUrl,0,16);
                (void)strncpy(appControlInfo.ipInfo[0].multicastUrl,argv[i+1],15);
                (void)memset(appControlInfo.ipInfo[1].multicastUrl,0,16);
                (void)strncpy(appControlInfo.ipInfo[1].multicastUrl,argv[i+1],15);
            }
            else
            {
                (void)printf("Warning : Missing multicast url address\n");
            }
        }
        else
        if ( !strcmp(argv[i], "-multicast_port") )
        {
            arg_increment++;
            if ((i+1)<argc)
            {
                appControlInfo.ipInfo[0].multicastPort = (uint32_t)atoi(argv[i+1]);
                appControlInfo.ipInfo[1].multicastPort = (uint32_t)atoi(argv[i+1]);
                if (appControlInfo.ipInfo[0].multicastPort > 65536)
                {
                    (void)printf("Warning : Invalid port range\n");
                    appControlInfo.ipInfo[0].multicastPort = (uint32_t) ((void *)0);
                    appControlInfo.ipInfo[1].multicastPort = (uint32_t) ((void *)0);
                }

            }
            else
            {
                (void)printf("Warning : Missing multicast port parameter\n");
            }
        }
        else
        if ( !strcmp(argv[i], "-seekfile") )
        {
            arg_increment++;
            if ((i+1) < argc)
            {
                appControlInfo.seekFd = fopen(argv[i+1], "r");
                if (appControlInfo.seekFd == ((void *)0))
                {
                    (void)printf("Warning : Unable to open seek file %s\n", argv[i+1]);
                }
            }
            else
            {
                (void)printf("Warning : Missing seek filename\n");
            }
        }
        else
        {
            (void)printf("Info : Unrecognised option ('%s') for %s - passing onto DirectFB.\n", argv[i], argv[0]);
        }
    }
}


static void signal_handler(int32_t thisSignal)
{
    static _Bool caught = 0;

    if (!caught)
    {
        caught = 1;
        (void)printf("Stopping Command Loop ... (signal %d)\n", thisSignal);
        keepCommandLoopAlive = 0;
    }
}

typedef struct _args {
    int32_t argc;
    char** argv;
}demoArgs;

static void setupFBResolution(char* fbName, int32_t width, int32_t height)
{
    int32_t fd;
    struct fb_var_screeninfo vinfo;
    int32_t error;
    fd = open(fbName, 02);

    if (fd >= 0)
    {

        error = ioctl(fd, 0x4600, &vinfo);
        if (error != 0)
        {
            (void)printf("Error reading variable information for '%s' - error code %d\n", fbName, error);
        }

        vinfo.xres = (uint32_t)width;
        vinfo.yres = (uint32_t)height;
        vinfo.activate = 128;


        error = ioctl(fd, 0x4601, &vinfo);
        if (error != 0)
        {
            (void)printf("Error setting variable information for '%s' - error code %d\n", fbName, error);
        }

        (void)close(fd);
    }
}
static void setupFramebuffers(void)
{
    if (appControlInfo.restartResolution)
    {
        int32_t fd;
        int32_t width;
        int32_t height;
        int32_t multipleFrameBuffers;

        multipleFrameBuffers = exStbDemo_fileExists("/dev/fb1");

        if (multipleFrameBuffers)
        {
            switch (appControlInfo.restartResolution & DSOR_ALL)
            {
                case(DSOR_720_480) :
                    {
                        width = 720;
                        height = 480;
                    }
                    break;
                case(DSOR_720_576) :
                    {
                        width = 720;
                        height = 576;
                    }
                    break;
                case(DSOR_1280_720) :
                    {
                        width = 720;
                        height = 720;
                    }
                    break;
                case(DSOR_1920_1080) :
                    {
                        width = 960;
                        height = 540;
                    }
                    break;
                default :
                    {
                        width = 0;
                        height = 0;
                    }
                    break;
            }
        }
        else
        {
            switch (appControlInfo.restartResolution & DSOR_ALL)
            {
                case(DSOR_720_480) :
                    {
                        width = 720;
                        height = 480;
                    }
                    break;
                case(DSOR_720_576) :
                    {
                        width = 720;
                        height = 576;
                    }
                    break;
                case(DSOR_1280_720) :
                    {
                        width = 1280;
                        height = 720;
                    }
                    break;
                case(DSOR_1920_1080) :
                    {
                        width = 1920;
                        if ((appControlInfo.restartResolution & (0x20000000u)) == (0x20000000u))
                        {
                            height = 1080;
                        }
                        else
                        {
                            height = 1023;
                        }
                    }
                    break;
                default :
                    {
                        width = 0;
                        height = 0;
                    }
                    break;
            }
        }


        fd = open("/etc/directfbrc", 02);
        if (fd >= 0)
        {
            char text[128];

            (void)sprintf(text, "mode=%dx%d\n", width, height);
            (void)write(fd, text, strlen(text));
            (void)strcpy(text, "pixelformat=ARGB\n");
            (void)write(fd, text, strlen(text));
            (void)strcpy(text, "depth=32\n");
            (void)write(fd, text, strlen(text));
            (void)strcpy(text, "no-vt-switch\n");
            (void)write(fd, text, strlen(text));
            (void)strcpy(text, "\n");
            (void)write(fd, text, strlen(text));

            (void)close(fd);
        }

        setupFBResolution("/dev/fb0", width, height);

        setupFBResolution("/dev/fb1", width, height);
    }
}

static void *main_Thread(void *pArg)
{
    demoArgs* pDemoArgs = (demoArgs*)pArg;
    DFBResult ret;

    exStbDemo_threadRename("Main");
    do
    {
        setupFramebuffers();

        dvb_init();

        gfx_init( pDemoArgs->argc, pDemoArgs->argv);

        (void)signal(2, signal_handler);
        (void)signal(15, signal_handler);
        (void)signal(11, signal_handler);

        ret = pgfx_dfb->CreateInputEventBuffer( pgfx_dfb, DICAPS_KEYS|DICAPS_BUTTONS|DICAPS_AXES, DFB_TRUE, &eventBuffer);

        if (ret)
        {
             (void)DirectFBError( "CreateInputEventBuffer() failed", ret );
             return ((void *)0);
        }

        if (slideShow_init() != 0)
        {
            (void)printf("Error starting slideshow thread\n");
        }

        if (sound_init() != 0)
        {
            (void)printf("Error starting sound implementation\n");
        }

        monitoring_init();

        menuApp_buildInitial();

        commandLoop(eventBuffer);
        eventBuffer = ((void *)0);

        monitoring_term();
        menuApp_terminate();
        (void)printf("Exiting DirectFB ...\n");
        gfx_terminate();
        (void)printf("Stopping DVB ...\n");
        dvb_terminate();

        pCurrentMenu = ((void *)0);
    }while(appControlInfo.restartGraphics);

    if (appControlInfo.commandInfo.inputFile)
    {
        (void)printf("Closing input command file ...\n");
        (void)close(appControlInfo.commandInfo.inputFile);
    }

    if (appControlInfo.commandInfo.outputFile)
    {
        (void)printf("Closing output log file ...\n");
        (void)close(appControlInfo.commandInfo.outputFile);
    }

    mainLoopActive = 0;

    return ((void *)0);
}

static void set_to_raw(int32_t tty_fd)
{
    struct termios tty_attr;

    (void)tcgetattr(tty_fd,&tty_attr);
    tty_attr.c_lflag &= ~((uint32_t)0000002);
    (void)tcsetattr(tty_fd,0,&tty_attr);
}

static void set_to_buffered(int32_t tty_fd)
{
    struct termios tty_attr;

    (void)tcgetattr(tty_fd,&tty_attr);
    tty_attr.c_lflag |= 0000002;
    (void)tcsetattr(tty_fd,0,&tty_attr);
}

int32_t main(int32_t argc, char* argv[])
{

__ESBMC_assume(argc>=0 && argc<(sizeof(argv)/sizeof(char)));
int counter;
for(counter=0; counter<argc; counter++)
  __ESBMC_assume(argv[counter]!=((void *)0));

    tmErrorCode_t audioStatus;
    int32_t status;
    pthread_t thread;
    demoArgs myArgs;
    DFBEvent event;
    int32_t term;




    do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0); do{}while(0);;


    event.clazz = DFEC_INPUT;
    event.input.type = DIET_KEYPRESS;
    event.input.flags = DIEF_KEYSYMBOL;
    event.input.key_symbol = DIKS_NULL;

    term = open("/dev/fd/0", 02);

    set_to_raw(term);


    while(!exStbDemo_fileExists("/dev/fb0"))
    {
        (void)sleep(1);
    }

    appInfo_init();

    checkCommandParams(argc, argv);

    myArgs.argc = argc;
    myArgs.argv = argv;

    audioStatus = tmdlAudioIO_OpenM( &gAIO, 0 );
    if(audioStatus != 0)
    {
        (void)fprintf(stderr, "exStbDemo: error in tmdlAudioIO_OpenM %d", audioStatus);

        return 1;
    }

    (void)tmdlAudioIO_SelectBeep(gAIO,3);



    status = pthread_create (&thread, ((void *)0),
                             main_Thread,
                             (void*)&myArgs);
    if (status != 0)
    {
        (void)fprintf(stderr, "exStbDemo: error in pthread_create %d", status);

        return 1;
    }

    while(eventBuffer == ((void *)0))
    {
        (void)usleep(100000);
    }


    if (appControlInfo.seekFd != ((void *)0))
    {
        _Bool firstTime = 1;
        char lineBuf[256];
        unsigned int sleepTime = 10;

        while (fgets(lineBuf, 256, appControlInfo.seekFd) != ((void *)0))
        {
            if (lineBuf[0] != '#')
            {
                if (!strncmp(lineBuf, "stop", strlen("stop")))
                {
                    (void)printf("Terminating seek file processing on 'stop' directive.\n");
                    break;
                }
                else if (!strncmp(lineBuf, "sleep", strlen("sleep")))
                {
                    int matches = sscanf(lineBuf, "sleep %d\n", &sleepTime);
                    if (matches >= 1)
                    {
                        (void)printf("Setting sleep time to %d.\n", sleepTime);
                    }
                }
                else if (firstTime)
                {
                    firstTime = 0;
                    char *pTemp;

                    pTemp = strcpy(appControlInfo.mediaInfo.filename, lineBuf);
                    pTemp[strcspn(appControlInfo.mediaInfo.filename, " \t\n")] = '\0';
                    (void)printf("Starting play back on %s\n", appControlInfo.mediaInfo.filename);
                    media_startPlayback();
                    (void)sleep(sleepTime);
                }
                else
                {
                    unsigned long int value;
                    char *pEndPtr;

                    value = strtoul(lineBuf, &pEndPtr, 0);
                    if (pEndPtr != lineBuf)
                    {
                        (void)printf("Seeking to byte offset %lu in file %s\n", value, appControlInfo.mediaInfo.filename);
                        media_pause();
                        media_seek( value );
                        media_pause();
                        (void)sleep(sleepTime);
                    }
                    else
                    {
                        (void)printf("Found invalid offset %s", lineBuf);
                    }
                }
            }
        }
    }

    while((event.input.key_symbol != DIKS_POWER) && keepCommandLoopAlive && (appControlInfo.commandInfo.inputFile == 0))
    {
        int32_t value;
        value = getchar();
        value = (int32_t)tolower(value);
        event.input.key_symbol = DIKS_NULL;
        switch (value)
        {
            case(0x1B) :
            {
                int32_t escapeValue[2];

                escapeValue[0] = getchar();
                escapeValue[1] = getchar();
                if (escapeValue[0] == 0x5B)
                {
                    switch (escapeValue[1])
                    {
                        case(0x41) : event.input.key_symbol = DIKS_CURSOR_UP; break;
                        case(0x42) : event.input.key_symbol = DIKS_CURSOR_DOWN; break;
                        case(0x43) : event.input.key_symbol = DIKS_CURSOR_RIGHT; break;
                        case(0x44) : event.input.key_symbol = DIKS_CURSOR_LEFT; break;
                        case(0x45) : event.input.key_symbol = DIKS_OK; break;
                        default : break;
                    }
                }
                break;
            }
            case('m') : event.input.key_symbol = DIKS_MENU; break;
            case('u') : event.input.key_symbol = DIKS_CURSOR_UP; break;
            case('d') : event.input.key_symbol = DIKS_CURSOR_DOWN; break;
            case('l') : event.input.key_symbol = DIKS_CURSOR_LEFT; break;
            case(0x0A): event.input.key_symbol = DIKS_CURSOR_RIGHT; break;
            case('r') : event.input.key_symbol = DIKS_CURSOR_RIGHT; break;
            case('+') : event.input.key_symbol = DIKS_VOLUME_UP; break;
            case('-') : event.input.key_symbol = DIKS_VOLUME_DOWN; break;
            case('s') : event.input.key_symbol = DIKS_MUTE; break;
            case('i') : event.input.key_symbol = (DFBInputDeviceKeySymbol)((DIKT_CUSTOM) | ((1))); break;
            case('0') : event.input.key_symbol = DIKS_0; break;
            case('1') : event.input.key_symbol = DIKS_1; break;
            case('2') : event.input.key_symbol = DIKS_2; break;
            case('3') : event.input.key_symbol = DIKS_3; break;
            case('4') : event.input.key_symbol = DIKS_4; break;
            case('5') : event.input.key_symbol = DIKS_5; break;
            case('6') : event.input.key_symbol = DIKS_6; break;
            case('7') : event.input.key_symbol = DIKS_7; break;
            case('8') : event.input.key_symbol = DIKS_8; break;
            case('9') : event.input.key_symbol = DIKS_9; break;
            case('x') : event.input.key_symbol = DIKS_POWER; break;
            case('q') : event.input.key_symbol = (DFBInputDeviceKeySymbol)((DIKT_CUSTOM) | ((14))); break;
            case('p') : event.input.key_symbol = DIKS_EXIT; break;
            case('o') : case('k') : event.input.key_symbol = DIKS_OK; break;
            default : break;
        }
        if (event.input.key_symbol != 0)
        {
            (void)printf("\n");
            event.input.flags = DIEF_KEYSYMBOL;
            if (eventBuffer)
            {
                eventBuffer->PostEvent(eventBuffer, &event);
            }
        }
    }

    (void)pthread_join (thread, ((void *)0));

    set_to_buffered(term);

    (void)close(term);
    (void)tmdlAudioIO_Close( gAIO );
    return 0;
}
