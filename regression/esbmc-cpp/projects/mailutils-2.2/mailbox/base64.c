/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2009, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <mailutils/errno.h>

int
mu_base64_encode (const unsigned char *input, size_t input_len,
		  unsigned char **output, size_t *output_len)
{
  static char b64tab[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  size_t olen = 4 * (input_len + 2) / 3 + 1;
  unsigned char *out = malloc (olen);

  if (!out)
    return ENOMEM;
  *output = out;
  while (input_len >= 3)
    {
      *out++ = b64tab[input[0] >> 2];
      *out++ = b64tab[((input[0] << 4) & 0x30) | (input[1] >> 4)];
      *out++ = b64tab[((input[1] << 2) & 0x3c) | (input[2] >> 6)];
      *out++ = b64tab[input[2] & 0x3f];
      input_len -= 3;
      input += 3;
    }

  if (input_len > 0)
    {
      unsigned char c = (input[0] << 4) & 0x30;
      *out++ = b64tab[input[0] >> 2];
      if (input_len > 1)
	c |= input[1] >> 4;
      *out++ = b64tab[c];
      *out++ = (input_len < 2) ? '=' : b64tab[(input[1] << 2) & 0x3c];
      *out++ = '=';
    }
  *output_len = out - *output;
  *out = 0;
  return 0;
}

int
mu_base64_decode (const unsigned char *input, size_t input_len,
		  unsigned char **output, size_t *output_len)
{
  static int b64val[128] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,
    -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1
  };
  int olen = input_len;
  unsigned char *out = malloc (olen);

  if (!out)
    return ENOMEM;
  *output = out;
  do
    {
      if (input[0] > 127 || b64val[input[0]] == -1
	  || input[1] > 127 || b64val[input[1]] == -1
	  || input[2] > 127 || ((input[2] != '=') && (b64val[input[2]] == -1))
	  || input[3] > 127 || ((input[3] != '=')
				&& (b64val[input[3]] == -1)))
	return -1;
      *out++ = (b64val[input[0]] << 2) | (b64val[input[1]] >> 4);
      if (input[2] != '=')
	{
	  *out++ = ((b64val[input[1]] << 4) & 0xf0) | (b64val[input[2]] >> 2);
	  if (input[3] != '=')
	    *out++ = ((b64val[input[2]] << 6) & 0xc0) | b64val[input[3]];
	}
      input += 4;
      input_len -= 4;
    }
  while (input_len > 0);
  *output_len = out - *output;
  return 0;
}

