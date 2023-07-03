# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

import time, sys
import fcntl
import struct

#------------------------------------------------------------------------------- 

def simple_increment(count, bDone=False, sleep=0.0):
    """                        
    A simple counter that prints count/total to console.
    sleep in seconds       
    """
    sys.stdout.write('\r' + bold_text('\t %d' % (count)))
    sys.stdout.flush()
    if sleep:
        time.sleep(sleep)
    if bDone:
        print((color('purple','\nDone     \n')))

def simple_counter(count,total,sleep=False):
    """                                              
    A simple counter that prints count/total to console.   
    """
    sys.stdout.write('\r' + bold_text('\t %d / %d' % (count+1, total)))
    sys.stdout.flush()
    if sleep:
        time.sleep(0.5)
    if count+1==total:
        print((color('purple','\nDone     \n')))

def color(color_val, string):
    """                          
    Set text color.                                                                                           

    Valid color values are: 
      30 or 'black' 
      31 or 'red'
      32 or 'green' 
      33 or 'yellow' 
      34 or 'blue' 
      35 or 'purple' 
      36 or 'periwinkle'
      37 or 'white' 
    """
    color_dict = {'black':'30', 'red':'31', 'green':'32', 'yellow':'33',
                  'blue':'34', 'purple':'35', 'periwinkle':'36', 'white':'37'}
    if type(color_val) == str:
        color_num = color_dict[color_val]
    elif type(color_val) == int:
        color_num = str(color_val)
    else:
        raise ValueError("Incorrect color value. See doc string.")
    return "\033[" + color_num + "m" + string + "\033[0m"

def bold_text(string):
    """ 
    Make text bold. 
    """
    return '\033[1m%s\033[0m' % string

def clear_screen():
    """
    Clear screen, return cursor to top left 
    """
    sys.stdout.write('\033[2J')
    sys.stdout.write('\033[H')
    sys.stdout.flush()

def progress(current, total):
    """  
    Generate progress bar data.       
    """
    try:
        import termios
        COLS = struct.unpack('hh',  fcntl.ioctl(sys.stdout,
                                                termios.TIOCGWINSZ, '1234'))[1]
    except:
        COLS = 50
    prefix = '        %d / %d' % (current, total)
    bar_start = ' ['
    bar_end = '] '

    bar_size = COLS - len(prefix + bar_start + bar_end)
    amount = int(current / (total / float(bar_size)))
    remain = bar_size - amount

    bar = 'X' * amount + ' ' * remain
    return bold_text(prefix) + bar_start + bar + bar_end

def simple_progress_bar(count, total, sleep=False):
    """              
    A simple progress bar the prints count/total and a progress bar to console.
    """
    sys.stdout.write('\r' + progress(count+1,total))
    sys.stdout.flush()
    if sleep:
        time.sleep(0.1)
    if count+1==total:
        print((color('purple','\nDone \n')))

        
if __name__ == '__main__':
    # Run a few simple tests:     
    # Test clear screen       
    clear_screen()

    # Test simple increment       
    print("A simple increment: \n")
    for i in range(10):
        simple_increment(i,sleep=0.5)
    simple_increment(i,bDone=True)

    # Test simple counter                                
    print("A simple counter: \n")
    for i in range(10):
        simple_counter(i,10,sleep=True)

    # Test colors                                   
    print("Colored text options: ")
    for i in range(30, 38):
        print((color(i, 'color '+str(i))))
    print('\n')

    # Test simple progress bar                                          
    print("A simple progress bar: \n")
    for i in range(100):
        simple_progress_bar(i,100,sleep=True)
