#include "SI_SC_Matrix_EMFields_Cyl3D.hxx"
#include "BaseFunctionDefine.hxx"
#include "PhysConsts.hxx"

void
SI_SC_Matrix_EMFields_Cyl3D::
    BuildFuncs()
{
    Standard_Real dt = this->GetDelTime();

    Standard_Integer dynEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex(); // 0
    Standard_Integer dynJIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_J_PhysDataIndex();                // 1
    Standard_Integer PreIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_PRE_PhysDataIndex();               // 4
    Standard_Integer AEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_AE_PhysDataIndex();                 // 2
    Standard_Integer BEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_BE_PhysDataIndex();                 // 3

    Standard_Integer dynHIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();  // 0
    Standard_Integer JMIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_JM_PhysDataIndex();                 // 1
    
    Standard_Integer PE1Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PE1_PhysDataIndex();          // 5
    Standard_Integer PE2Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PE2_PhysDataIndex();          // 6

    Standard_Integer PM1Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PM1_PhysDataIndex();          // 2
    Standard_Integer PM2Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PM2_PhysDataIndex();          // 3

    // add 
    Standard_Integer preTStepEFldIndx = GetFldsDefCntr()->GetFieldsDefineRules()->Get_MUR_PreStep_PhysDataIndex();

    if(1){
        std::cout << "dynEIndex = " << dynEIndex << "\n";
        std::cout << "dynJIndex = " << dynJIndex << "\n";
        std::cout << "PreIndex = " << PreIndex << "\n";
        std::cout << "AEIndex = " << AEIndex << "\n";
        std::cout << "BEIndex = " << BEIndex << "\n";
        std::cout << "dynHIndex = " << dynHIndex << "\n";
        std::cout << "JMIndex = " << JMIndex << "\n";
        std::cout << "PE1Index = " << PE1Index << "\n";
        std::cout << "PE2Index = " << PE2Index << "\n";
        std::cout << "PM1Index = " << PM1Index << "\n";
        std::cout << "PM2Index = " << PM2Index << "\n";
        std::cout << "preTStepEFldIndx = " << preTStepEFldIndx << "\n";
    }

    bool doDamping;
    if(m_Damping > 0.001){
        doDamping = true;
    }
    else{
        doDamping = false;
    }
    
    Build_Elec_Func(doDamping, dt, dynEIndex, dynJIndex, PreIndex, AEIndex, BEIndex, dynHIndex);
    Build_Elec_CPML_Func(doDamping, dt, dynEIndex, PreIndex, AEIndex, BEIndex, dynHIndex, PE1Index, PE2Index);
    
    Build_Elec_Mur_Func(doDamping, dt, dynEIndex, PreIndex, AEIndex, BEIndex, preTStepEFldIndx);

    if(doDamping){
        Build_Mag_Func(dt, dynHIndex, JMIndex, BEIndex);
        Build_Mag_CPML_Func(dt, dynHIndex, PM1Index, PM2Index, BEIndex);
    }
    else{
        Build_Mag_Func(dt, dynHIndex, JMIndex, dynEIndex);
        Build_Mag_CPML_Func(dt, dynHIndex, PM1Index, PM2Index, dynEIndex);
    }
}