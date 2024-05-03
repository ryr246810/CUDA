//--------------------------------------------------------------------
//
// File:        Txbase.cpp
//
// Purpose:     WINDOWS only Defines the entry point for the DLL application.
//
// Copyright (c) 2006 Tech-X Corporation
//
// All rights reserved.
//
// Version:     $Id: TxBase.cpp 52 2006-07-26 21:20:47Z yew $
//
//--------------------------------------------------------------------

//

#ifdef _TXOPTSOLVE_DLL

#include <TxBase.h>

BOOL APIENTRY DllMain( HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved )
{
  switch (ul_reason_for_call) 
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
      break;
    }
  return TRUE;
}

#endif

