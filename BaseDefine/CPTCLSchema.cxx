/**
 * @file  CPTCLSchema.cpp
 *
 * @class CPTCLSchema
 *
 * @brief Describes how instant datasets and meshes are found in the
 * file compliant with the schema.
 *
 */

#include <CPTCLSchema.hxx>

std::string CPTCLSchema::typeAtt ="CPTCLType";


std::string CPTCLSchema::numSpatialDimsAtt = "numSpatialDims";


// dataset Labels
std::string CPTCLSchema::labelsAtt = "labels";
std::string CPTCLSchema::labelsKey = "x,y,z,vx,vy,vz";

std::string CPTCLSchema::velocityAtt = "velocity";
std::string CPTCLSchema::velocityKey = "{vx,vy,vz}";

std::string CPTCLSchema::varKey = "variable";
std::string CPTCLSchema::varsKey = "vars";
std::string CPTCLSchema::varWithMeshKey = "variableWithMesh";



// centering key
std::string CPTCLSchema::zonalCenteringKey = "zonal"; 
std::string CPTCLSchema::nodalCenteringKey = "nodal"; // Default
std::string CPTCLSchema::edgeCenteringKey = "edge";
std::string CPTCLSchema::faceCenteringKey = "face";

// Index ordering...
std::string CPTCLSchema::compMajorCKey = "compMajorC"; //currently not supported
std::string CPTCLSchema::compMinorCKey = "compMinorC"; //default ordering
std::string CPTCLSchema::compMajorFKey = "compMajorF"; //currently not supported
std::string CPTCLSchema::compMinorFKey = "compMinorF"; //supported


//Time
std::string CPTCLSchema::timeKey = "time";
std::string CPTCLSchema::timeAtt = "Time";
std::string CPTCLSchema::cycleAtt = "Step";



//Run info
std::string CPTCLSchema::runInfoKey = "runInfo";
std::string CPTCLSchema::softwareAtt = "CEMPICSoftware";

