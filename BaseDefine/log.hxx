/*********************************************************
COPYRIGHT(C), 
File NAME:   log.hxx
Author:      softwind
Version:     1.0
CreateDate:  2020-04-29 14:02
Description: 

Modify History:

*********************************************************/

#ifndef _LOG__
#define _LOG__

#define DEBUG_ON
#define ERROR_ON
#define WARNING_ON
#define INFO_ON
#define NORMAL_ON

#ifdef _WIN32
#include <windows.h>
using namespace std;
    #define NONE         SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)
    #define RED          SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED)
    #define LIGHT_RED    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED)
    #define GREEN        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_GREEN)
    #define LIGHT_GREEN  SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_BLUE)
    #define BLUE         SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_BLUE)
    #define LIGHT_BLUE   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_BLUE)
    #define DARY_GRAY    
    #define CYAN         SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN | FOREGROUND_BLUE)
    #define LIGHT_CYAN   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY  | FOREGROUND_GREEN | FOREGROUND_BLUE)
    #define PURPLE       SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_BLUE)
    #define LIGHT_PURPLE SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_BLUE)
    #define BROWN        
    #define YELLOW       SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN)
    #define LIGHT_GRAY   
    #define WHITE        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)
    #define LIGHT_WHITE  SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)
    #define END          "\n"

    #ifdef DEBUG_ON
    #define DEBUG(format, ...) \
        do \
        { \
            YELLOW; \
            printf (format END, ##__VA_ARGS__); \
            NONE; \
        } while (0)
    #else
    #define DEBUG(format, ...)
    #endif

    #ifdef NORMAL_ON
    #define LOG_PRINT(format, ...) \
        do \
        { \
            NONE;  \
            printf(format END, ##__VA_ARGS__); \
        } while (0)
    #else
    #define LOG_PRINT(format, ...)
    #endif

    #ifdef ERROR_ON
    #define ERR_PRINT(format, ...) \
        do \
        { \
            RED;  \
            fprintf(stderr, "Error ------>[file: %s line: %d] ", __FILE__, __LINE__); \
            fprintf(stderr, format END, ##__VA_ARGS__); \
            NONE; \
        } while (0)
    #else
    #define ERR_PRINT(format, ...)
    #endif

    #ifdef WARNING_ON
    #define WARNING_PRINT(format, ...) \
        do \
        { \
            LIGHT_PURPLE;  \
            printf("Warning ------> [file: %s line: %d] ", __FILE__, __LINE__); \
            printf(format END, ##__VA_ARGS__); \
            NONE; \
        } while (0)
    #else
    #define WARNING_PRINT(format, ...)
    #endif

    #ifdef INFO_ON
    #define INFO_PRINT(format, ...) \
        do \
        { \
            LIGHT_GREEN; \
            printf(format END, ##__VA_ARGS__); \
            NONE; \
        } while(0)
    #else
    #define INFO_PRINT(format, ...)
    #endif

#else
    #define NONE         "\033[m"
    #define RED          "\033[0;32;31m"
    #define LIGHT_RED    "\033[1;31m"
    #define GREEN        "\033[0;32;32m"
    #define LIGHT_GREEN  "\033[1;32m"
    #define BLUE         "\033[0;32;34m"
    #define LIGHT_BLUE   "\033[1;34m"
    #define DARY_GRAY    "\033[1;30m"
    #define CYAN         "\033[0;36m"
    #define LIGHT_CYAN   "\033[1;36m"
    #define PURPLE       "\033[0;35m"
    #define LIGHT_PURPLE "\033[1;35m"
    #define BROWN        "\033[0;33m"
    #define YELLOW       "\033[1;33m"
    #define LIGHT_GRAY   "\033[0;37m"
    #define WHITE        "\033[1;37m"
    #define END          "\033[0m \n"

    //fprintf(stderr, RED " Error: %s \n", "No waveLength or frequency is defined!\033[0m");

    #ifdef DEBUG_ON
    #define DEBUG(format, ...) printf (YELLOW format END, ##__VA_ARGS__)
    #else
    #define DEBUG(format, ...)
    #endif

    #ifdef NORMAL_ON
    #define LOG_PRINT(format, ...) \
        do \
        { \
            printf(format, ##__VA_ARGS__); \
            printf(END); \
        } while (0)
    #else
    #define LOG_PRINT(format, ...)
    #endif

    #ifdef ERROR_ON
    #define ERR_PRINT(format, ...) \
        do \
        { \
            fprintf(stderr, LIGHT_RED); \
            fprintf(stderr, "Error ------>[file: %s line: %d] ", __FILE__, __LINE__); \
            fprintf(stderr, format, ##__VA_ARGS__); \
            fprintf(stderr, END); \
        } while (0)
    #else
    #define ERR_PRINT(str)
    #endif

    #ifdef WARNING_ON
    #define WARNING_PRINT(format, ...) \
        do \
        { \
            printf(LIGHT_PURPLE); \
            printf("Warning ------> [file: %s line: %d] ", __FILE__, __LINE__); \
            printf(format, ##__VA_ARGS__); \
            printf(END); \
        } while (0)
    #else
    #define WARNING_PRINT(str)
    #endif

    #ifdef INFO_ON
    #define INFO_PRINT(format, ...) \
        do \
        { \
            printf(LIGHT_GREEN); \
            printf(format, ##__VA_ARGS__); \
            printf(END); \
        } while(0)
    #else
    #define INFO_PRINT(str)
    #endif
#endif


#endif