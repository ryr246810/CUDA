//--------------------------------------------------------------------
//
// File:        Txbase.h 
//
// Purpose:     This file contains a VC++ - specific macro.
//              
//
// Copyright (c) 2006 Tech-X Corporation
//
// All rights reserved.
//
// Version:     $Id: TxBase.h 30 2006-04-21 20:42:51Z yew $ 
//
//--------------------------------------------------------------------

//This file contains a VC++ - specific macro.
// The macro defines to nothing for any other compiler
// However, this Macro has to be added to any new class that is exported outside of the dll boundaries

// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the TXBASE_EXPORTS
// symbol defined on the command line (or in the .dsp file).  
// This way any other project whose source files include this file see 
// TXBASE_API functions as being imported from a DLL, wheras this DLL sees symbols
// defined with this macro as being exported.

#ifdef _TXOPTSOLVE_DLL

#include <Windows.h>

#ifdef TXBASE_EXPORTS
#define TXBASE_API __declspec(dllexport)
#else
#define TXBASE_API __declspec(dllimport)
#endif

BOOL APIENTRY DllMain( HANDLE hModule,  DWORD  ul_reason_for_call,  LPVOID lpReserved  );

// This VC++6.0 warning says that all data members and parent classses
// need to be exported, if a given class is exported.
// It is not a problem as long as correct stl headers are used
// But the compiler needs to be pacified 
// #pragma warning (disable:4251)

#else

#define TXBASE_API

#endif

/* Example of using 
// This class is exported from the txbase.dll
class TXBASE_API CTxbase 
{
public:
	CTxbase(void);
	// TODO: add your methods here.
};

extern TXBASE_API int nTxbasel;

TXBASE_API int fnTxbase(void);
*/

