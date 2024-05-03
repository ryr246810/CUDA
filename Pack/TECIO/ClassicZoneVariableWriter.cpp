#include "ClassicZoneVariableWriter.h"
#include "ThirdPartyHeadersBegin.h"
#include <boost/scoped_array.hpp>
#include <stdexcept>
#include "ThirdPartyHeadersEnd.h"
#include "AltTecUtil.h"
#include "FieldData.h"
#include "fileStuff.h"
#include "FileWriterInterface.h"
#include "ItemSetIterator.h"
#include "writeValueArray.h"
namespace tecplot { namespace ___3933 { namespace { char const* CC_FIELD_DATA_MARKER_LABEL = "ccFieldDataMarker*"; char const* NODAL_FIELD_DATA_MARKER_LABEL = "nodalFieldDataMarker*"; char const* VARIABLE_NUM_LABEL = "variableNum*"; char const* FIELD_DATA_LABEL = "FieldData"; } ClassicZoneVariableWriter::ClassicZoneVariableWriter( ItemSetIterator& varIter, ___4636      zone, ___4636      ___341, ___37&      ___36) : m_varIter(varIter) , ___2677(zone) , m_baseZone(___341) , ___2337(___36) , m_zoneNumberLabel(SZPLT_ZONE_NUM_DESCRIPTION) {} uint64_t ClassicZoneVariableWriter::varHeaderSizeInFile(bool ___2002) { if (___2002) return 3 * valueSizeInFile<uint32_t, false>(___2002); return 0; } uint64_t ClassicZoneVariableWriter::varSizeInFile(___4352 ___4336, bool ___2002) const { uint64_t ___3358 = varHeaderSizeInFile(___2002); ___1352 ___1351(&___2337, ___2677 + 1, ___4336 + 1); if (___1351.___2067()) { switch (___1351.getValueType()) { case FieldDataType_Float: ___3358 += arraySizeInFile<float, false>(static_cast<size_t>(___1351.___1783()), ___2002); break; case FieldDataType_Double: ___3358 += arraySizeInFile<double, false>(static_cast<size_t>(___1351.___1783()), ___2002); break; case FieldDataType_Int32: ___3358 += arraySizeInFile<int32_t, false>(static_cast<size_t>(___1351.___1783()), ___2002); break; case FieldDataType_Int16: ___3358 += arraySizeInFile<int16_t, false>(static_cast<size_t>(___1351.___1783()), ___2002); break; case FieldDataType_Byte: ___3358 += arraySizeInFile<uint8_t, false>(static_cast<size_t>(___1351.___1783()), ___2002); break; case ___1365: ___3358 += arraySizeInFile<uint8_t, false>(static_cast<size_t>(numBytesForNumBits(___1351.___1783())), ___2002); break; default: throw std::runtime_error("Unsupported variable type."); break; } } return ___3358; } ___372 ClassicZoneVariableWriter::writeVarHeader( FileWriterInterface& file, ValueLocation_e      ___4326, ___4352           ___4336) { ___372 ___2039 = ___4226; REQUIRE(file.___2041()); REQUIRE(___4326 == ___4328 || ___4326 == ___4330); if (file.___2002()) { if ( ___4326 == ___4328 ) ___2039 = ___2039 && writeValue<uint32_t, false, 0>(file, CC_FIELD_DATA_MARKER_LABEL, SZPLT_CC_FIELD_DATA_MARKER); else ___2039 = ___2039 && writeValue<uint32_t, false, 0>(file, NODAL_FIELD_DATA_MARKER_LABEL, SZPLT_NODAL_FIELD_DATA_MARKER); ___2039 = ___2039 && writeValue<uint32_t, false, 0>(file, VARIABLE_NUM_LABEL, (___4336 - m_varIter.baseItem() + 1)); ___2039 = ___2039 && writeValue<uint32_t, false, 0>(file, m_zoneNumberLabel.c_str(), (___2677 - m_baseZone + 1)); } ENSURE(VALID_BOOLEAN(___2039)); return ___2039; } namespace { template <typename T, bool isBitArray> ___372 writeFieldData( FileWriterInterface& szpltFile, ___1352 const&     ___1351) { ___372 ___2039 = ___4226; size_t ___4325 = static_cast<size_t>(___1351.___1783()); void* rawPointer = ___1351.getRawPointer(); if (rawPointer) { ___2039 = ___4563<T, false, 0>(szpltFile, FIELD_DATA_LABEL, ___2745, ___4325, (T*)rawPointer); } else { boost::scoped_array<T> array; array.reset(new T[___4325]); if (___2039) { for (size_t i = 0; i < ___4325; ++i) { double ___4298 = ___1351.___1780(static_cast<___2227>(i) + 1); array[i] = static_cast<T>(___4298); } ___2039 = ___2039 && ___4563<T, false, 0>(szpltFile, FIELD_DATA_LABEL, ___2745, ___4325, &array[0]); } } ENSURE(VALID_BOOLEAN(___2039)); return ___2039; } template <> ___372 writeFieldData<uint8_t, true>( FileWriterInterface& szpltFile, ___1352 const&     ___1351) { REQUIRE(___1351.getValueType() == ___1365); ___372 ___2039 = ___4226; ___1352 tempFieldData; size_t ___4325 = static_cast<size_t>(___1351.___1783()); size_t ___2779 = numBytesForNumBits(___4325); void* rawPointer = ___1351.getRawPointer(); if (!rawPointer) { tempFieldData.allocate(___1351.getValueType(), ___1351.___1786(), ___1351.___1783()); rawPointer = tempFieldData.getRawPointer(); if (rawPointer == NULL) throw std::bad_alloc(); for (___2227 i = 1; i <= ___1351.___1783(); ++i) tempFieldData.___3504(i, ___1351.___1780(i)); } ___2039 = ___2039 && ___4563<uint8_t, false, 0>(szpltFile, FIELD_DATA_LABEL, ___2745, ___2779, (uint8_t*)rawPointer); ENSURE(VALID_BOOLEAN(___2039)); return ___2039; } } ___372 ClassicZoneVariableWriter::writeVar( FileWriterInterface& szpltFile, ___1352 const&     ___1351) { ___372 ___2039 = ___4226; switch (___1351.getValueType()) { case FieldDataType_Float: ___2039 = writeFieldData<float, false>(szpltFile, ___1351); break; case FieldDataType_Double: ___2039 = writeFieldData<double, false>(szpltFile, ___1351); break; case FieldDataType_Int32: ___2039 = writeFieldData<int32_t, false>(szpltFile, ___1351); break; case FieldDataType_Int16: ___2039 = writeFieldData<int16_t, false>(szpltFile, ___1351); break; case FieldDataType_Byte: ___2039 = writeFieldData<uint8_t, false>(szpltFile, ___1351);
break; case ___1365: ___2039 = writeFieldData<uint8_t, true>(szpltFile, ___1351); break; default: throw std::runtime_error("Unsupported variable type."); break; } ENSURE(VALID_BOOLEAN(___2039)); return ___2039; } }}