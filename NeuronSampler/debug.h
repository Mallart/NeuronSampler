//
// Created by matha on 04/12/2024.
//

#ifndef DEBUG_H
#define DEBUG_H
#define MEM_TEST free(malloc(4096));

enum NS_ERROR
{
  // When you're pointing on something that isn't yours (please don't).
  EXCEPTION_INVALID_POINTER,
  // When you're out of your array bounds (not-so-smart thing to do).
  EXCEPTION_OUT_OF_BOUNDS,
  // Allocation couldn't end successfully.
  EXCEPTION_NOT_ENOUGH_MEMORY,
  // Stack is overloaded or corrupted
  EXCEPTION_STACK_CORRUPTION,
};

#define throw(e) { printf("A fatal error has occured and led the program to a crash.\nERROR_CODE: \t%i\n", e); exit(e); }

#endif //DEBUG_H
