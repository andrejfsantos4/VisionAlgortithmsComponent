# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vision_algorithms.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='vision_algorithms.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x17vision_algorithms.proto\"\x1d\n\x0c\x46loat1DArray\x12\r\n\x05\x65lems\x18\x01 \x03(\x02\",\n\x0c\x46loat2DArray\x12\x1c\n\x05lines\x18\x01 \x03(\x0b\x32\r.Float1DArray\"/\n\x0c\x46loat3DArray\x12\x1f\n\x08matrices\x18\x01 \x03(\x0b\x32\r.Float2DArray\"B\n\x05Image\x12\x1a\n\x03img\x18\x01 \x01(\x0b\x32\r.Float3DArray\x12\r\n\x05width\x18\x02 \x01(\x05\x12\x0e\n\x06height\x18\x03 \x01(\x05\"\xc5\x02\n\x0b\x45xecRequest\x12\x1f\n\x08klt_args\x18\x01 \x01(\x0b\x32\x0b.KltRequestH\x00\x12,\n\x0fhomog_calc_args\x18\x02 \x01(\x0b\x32\x11.HomogCalcRequestH\x00\x12,\n\x0fhomog_warp_args\x18\x03 \x01(\x0b\x32\x11.HomogWarpRequestH\x00\x12(\n\rsift_det_args\x18\x04 \x01(\x0b\x32\x0f.SiftDetRequestH\x00\x12,\n\x0fsift_match_args\x18\x05 \x01(\x0b\x32\x11.SiftMatchRequestH\x00\x12(\n\rline_det_args\x18\x06 \x01(\x0b\x32\x0f.LineDetRequestH\x00\x12/\n\x11vid_to_frame_args\x18\x07 \x01(\x0b\x32\x12.VidToFrameRequestH\x00\x42\x06\n\x04\x61rgs\"\xc6\x02\n\x0c\x45xecResponse\x12\x1f\n\x07klt_out\x18\x01 \x01(\x0b\x32\x0c.KltResponseH\x00\x12,\n\x0ehomog_calc_out\x18\x02 \x01(\x0b\x32\x12.HomogCalcResponseH\x00\x12,\n\x0ehomog_warp_out\x18\x03 \x01(\x0b\x32\x12.HomogWarpResponseH\x00\x12(\n\x0csift_det_out\x18\x04 \x01(\x0b\x32\x10.SiftDetResponseH\x00\x12,\n\x0esift_match_out\x18\x05 \x01(\x0b\x32\x12.SiftMatchResponseH\x00\x12(\n\x0cline_det_out\x18\x06 \x01(\x0b\x32\x10.LineDetResponseH\x00\x12/\n\x10vid_to_frame_out\x18\x07 \x01(\x0b\x32\x13.VidToFrameResponseH\x00\x42\x06\n\x04\x61rgs\"\x1c\n\nKltRequest\x12\x0e\n\x06\x62labla\x18\x01 \x01(\x02\"\x1d\n\x0bKltResponse\x12\x0e\n\x06\x62labla\x18\x01 \x01(\x02\"\x87\x01\n\x10HomogCalcRequest\x12!\n\nsource_pts\x18\x01 \x01(\x0b\x32\r.Float2DArray\x12\x1f\n\x08\x64\x65st_pts\x18\x02 \x01(\x0b\x32\r.Float2DArray\x12\x15\n\rransac_thresh\x18\x03 \x01(\x02\x12\x18\n\x10max_ransac_iters\x18\x04 \x01(\x05\"6\n\x11HomogCalcResponse\x12!\n\nhomography\x18\x01 \x01(\x0b\x32\r.Float2DArray\"s\n\x10HomogWarpRequest\x12\x15\n\x05image\x18\x01 \x01(\x0b\x32\x06.Image\x12!\n\nhomography\x18\x02 \x01(\x0b\x32\r.Float2DArray\x12\x11\n\tout_width\x18\x03 \x01(\x05\x12\x12\n\nout_height\x18\x04 \x01(\x05\"*\n\x11HomogWarpResponse\x12\x15\n\x05image\x18\x01 \x01(\x0b\x32\x06.Image\"\x10\n\x0eSiftDetRequest\"\x11\n\x0fSiftDetResponse\"\x12\n\x10SiftMatchRequest\"\x13\n\x11SiftMatchResponse\"\x10\n\x0eLineDetRequest\"\x11\n\x0fLineDetResponse\"\x13\n\x11VidToFrameRequest\"\x14\n\x12VidToFrameResponse2:\n\x10VisionAlgorithms\x12&\n\x07Process\x12\x0c.ExecRequest\x1a\r.ExecResponseb\x06proto3'
)




_FLOAT1DARRAY = _descriptor.Descriptor(
  name='Float1DArray',
  full_name='Float1DArray',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='elems', full_name='Float1DArray.elems', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27,
  serialized_end=56,
)


_FLOAT2DARRAY = _descriptor.Descriptor(
  name='Float2DArray',
  full_name='Float2DArray',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='lines', full_name='Float2DArray.lines', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=58,
  serialized_end=102,
)


_FLOAT3DARRAY = _descriptor.Descriptor(
  name='Float3DArray',
  full_name='Float3DArray',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='matrices', full_name='Float3DArray.matrices', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=104,
  serialized_end=151,
)


_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='img', full_name='Image.img', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width', full_name='Image.width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='height', full_name='Image.height', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=153,
  serialized_end=219,
)


_EXECREQUEST = _descriptor.Descriptor(
  name='ExecRequest',
  full_name='ExecRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='klt_args', full_name='ExecRequest.klt_args', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='homog_calc_args', full_name='ExecRequest.homog_calc_args', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='homog_warp_args', full_name='ExecRequest.homog_warp_args', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sift_det_args', full_name='ExecRequest.sift_det_args', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sift_match_args', full_name='ExecRequest.sift_match_args', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='line_det_args', full_name='ExecRequest.line_det_args', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vid_to_frame_args', full_name='ExecRequest.vid_to_frame_args', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='args', full_name='ExecRequest.args',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=222,
  serialized_end=547,
)


_EXECRESPONSE = _descriptor.Descriptor(
  name='ExecResponse',
  full_name='ExecResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='klt_out', full_name='ExecResponse.klt_out', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='homog_calc_out', full_name='ExecResponse.homog_calc_out', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='homog_warp_out', full_name='ExecResponse.homog_warp_out', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sift_det_out', full_name='ExecResponse.sift_det_out', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sift_match_out', full_name='ExecResponse.sift_match_out', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='line_det_out', full_name='ExecResponse.line_det_out', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vid_to_frame_out', full_name='ExecResponse.vid_to_frame_out', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='args', full_name='ExecResponse.args',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=550,
  serialized_end=876,
)


_KLTREQUEST = _descriptor.Descriptor(
  name='KltRequest',
  full_name='KltRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='blabla', full_name='KltRequest.blabla', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=878,
  serialized_end=906,
)


_KLTRESPONSE = _descriptor.Descriptor(
  name='KltResponse',
  full_name='KltResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='blabla', full_name='KltResponse.blabla', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=908,
  serialized_end=937,
)


_HOMOGCALCREQUEST = _descriptor.Descriptor(
  name='HomogCalcRequest',
  full_name='HomogCalcRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='source_pts', full_name='HomogCalcRequest.source_pts', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dest_pts', full_name='HomogCalcRequest.dest_pts', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ransac_thresh', full_name='HomogCalcRequest.ransac_thresh', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='max_ransac_iters', full_name='HomogCalcRequest.max_ransac_iters', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=940,
  serialized_end=1075,
)


_HOMOGCALCRESPONSE = _descriptor.Descriptor(
  name='HomogCalcResponse',
  full_name='HomogCalcResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='homography', full_name='HomogCalcResponse.homography', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1077,
  serialized_end=1131,
)


_HOMOGWARPREQUEST = _descriptor.Descriptor(
  name='HomogWarpRequest',
  full_name='HomogWarpRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='image', full_name='HomogWarpRequest.image', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='homography', full_name='HomogWarpRequest.homography', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='out_width', full_name='HomogWarpRequest.out_width', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='out_height', full_name='HomogWarpRequest.out_height', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1133,
  serialized_end=1248,
)


_HOMOGWARPRESPONSE = _descriptor.Descriptor(
  name='HomogWarpResponse',
  full_name='HomogWarpResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='image', full_name='HomogWarpResponse.image', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1250,
  serialized_end=1292,
)


_SIFTDETREQUEST = _descriptor.Descriptor(
  name='SiftDetRequest',
  full_name='SiftDetRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1294,
  serialized_end=1310,
)


_SIFTDETRESPONSE = _descriptor.Descriptor(
  name='SiftDetResponse',
  full_name='SiftDetResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1312,
  serialized_end=1329,
)


_SIFTMATCHREQUEST = _descriptor.Descriptor(
  name='SiftMatchRequest',
  full_name='SiftMatchRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1331,
  serialized_end=1349,
)


_SIFTMATCHRESPONSE = _descriptor.Descriptor(
  name='SiftMatchResponse',
  full_name='SiftMatchResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1351,
  serialized_end=1370,
)


_LINEDETREQUEST = _descriptor.Descriptor(
  name='LineDetRequest',
  full_name='LineDetRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1372,
  serialized_end=1388,
)


_LINEDETRESPONSE = _descriptor.Descriptor(
  name='LineDetResponse',
  full_name='LineDetResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1390,
  serialized_end=1407,
)


_VIDTOFRAMEREQUEST = _descriptor.Descriptor(
  name='VidToFrameRequest',
  full_name='VidToFrameRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1409,
  serialized_end=1428,
)


_VIDTOFRAMERESPONSE = _descriptor.Descriptor(
  name='VidToFrameResponse',
  full_name='VidToFrameResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1430,
  serialized_end=1450,
)

_FLOAT2DARRAY.fields_by_name['lines'].message_type = _FLOAT1DARRAY
_FLOAT3DARRAY.fields_by_name['matrices'].message_type = _FLOAT2DARRAY
_IMAGE.fields_by_name['img'].message_type = _FLOAT3DARRAY
_EXECREQUEST.fields_by_name['klt_args'].message_type = _KLTREQUEST
_EXECREQUEST.fields_by_name['homog_calc_args'].message_type = _HOMOGCALCREQUEST
_EXECREQUEST.fields_by_name['homog_warp_args'].message_type = _HOMOGWARPREQUEST
_EXECREQUEST.fields_by_name['sift_det_args'].message_type = _SIFTDETREQUEST
_EXECREQUEST.fields_by_name['sift_match_args'].message_type = _SIFTMATCHREQUEST
_EXECREQUEST.fields_by_name['line_det_args'].message_type = _LINEDETREQUEST
_EXECREQUEST.fields_by_name['vid_to_frame_args'].message_type = _VIDTOFRAMEREQUEST
_EXECREQUEST.oneofs_by_name['args'].fields.append(
  _EXECREQUEST.fields_by_name['klt_args'])
_EXECREQUEST.fields_by_name['klt_args'].containing_oneof = _EXECREQUEST.oneofs_by_name['args']
_EXECREQUEST.oneofs_by_name['args'].fields.append(
  _EXECREQUEST.fields_by_name['homog_calc_args'])
_EXECREQUEST.fields_by_name['homog_calc_args'].containing_oneof = _EXECREQUEST.oneofs_by_name['args']
_EXECREQUEST.oneofs_by_name['args'].fields.append(
  _EXECREQUEST.fields_by_name['homog_warp_args'])
_EXECREQUEST.fields_by_name['homog_warp_args'].containing_oneof = _EXECREQUEST.oneofs_by_name['args']
_EXECREQUEST.oneofs_by_name['args'].fields.append(
  _EXECREQUEST.fields_by_name['sift_det_args'])
_EXECREQUEST.fields_by_name['sift_det_args'].containing_oneof = _EXECREQUEST.oneofs_by_name['args']
_EXECREQUEST.oneofs_by_name['args'].fields.append(
  _EXECREQUEST.fields_by_name['sift_match_args'])
_EXECREQUEST.fields_by_name['sift_match_args'].containing_oneof = _EXECREQUEST.oneofs_by_name['args']
_EXECREQUEST.oneofs_by_name['args'].fields.append(
  _EXECREQUEST.fields_by_name['line_det_args'])
_EXECREQUEST.fields_by_name['line_det_args'].containing_oneof = _EXECREQUEST.oneofs_by_name['args']
_EXECREQUEST.oneofs_by_name['args'].fields.append(
  _EXECREQUEST.fields_by_name['vid_to_frame_args'])
_EXECREQUEST.fields_by_name['vid_to_frame_args'].containing_oneof = _EXECREQUEST.oneofs_by_name['args']
_EXECRESPONSE.fields_by_name['klt_out'].message_type = _KLTRESPONSE
_EXECRESPONSE.fields_by_name['homog_calc_out'].message_type = _HOMOGCALCRESPONSE
_EXECRESPONSE.fields_by_name['homog_warp_out'].message_type = _HOMOGWARPRESPONSE
_EXECRESPONSE.fields_by_name['sift_det_out'].message_type = _SIFTDETRESPONSE
_EXECRESPONSE.fields_by_name['sift_match_out'].message_type = _SIFTMATCHRESPONSE
_EXECRESPONSE.fields_by_name['line_det_out'].message_type = _LINEDETRESPONSE
_EXECRESPONSE.fields_by_name['vid_to_frame_out'].message_type = _VIDTOFRAMERESPONSE
_EXECRESPONSE.oneofs_by_name['args'].fields.append(
  _EXECRESPONSE.fields_by_name['klt_out'])
_EXECRESPONSE.fields_by_name['klt_out'].containing_oneof = _EXECRESPONSE.oneofs_by_name['args']
_EXECRESPONSE.oneofs_by_name['args'].fields.append(
  _EXECRESPONSE.fields_by_name['homog_calc_out'])
_EXECRESPONSE.fields_by_name['homog_calc_out'].containing_oneof = _EXECRESPONSE.oneofs_by_name['args']
_EXECRESPONSE.oneofs_by_name['args'].fields.append(
  _EXECRESPONSE.fields_by_name['homog_warp_out'])
_EXECRESPONSE.fields_by_name['homog_warp_out'].containing_oneof = _EXECRESPONSE.oneofs_by_name['args']
_EXECRESPONSE.oneofs_by_name['args'].fields.append(
  _EXECRESPONSE.fields_by_name['sift_det_out'])
_EXECRESPONSE.fields_by_name['sift_det_out'].containing_oneof = _EXECRESPONSE.oneofs_by_name['args']
_EXECRESPONSE.oneofs_by_name['args'].fields.append(
  _EXECRESPONSE.fields_by_name['sift_match_out'])
_EXECRESPONSE.fields_by_name['sift_match_out'].containing_oneof = _EXECRESPONSE.oneofs_by_name['args']
_EXECRESPONSE.oneofs_by_name['args'].fields.append(
  _EXECRESPONSE.fields_by_name['line_det_out'])
_EXECRESPONSE.fields_by_name['line_det_out'].containing_oneof = _EXECRESPONSE.oneofs_by_name['args']
_EXECRESPONSE.oneofs_by_name['args'].fields.append(
  _EXECRESPONSE.fields_by_name['vid_to_frame_out'])
_EXECRESPONSE.fields_by_name['vid_to_frame_out'].containing_oneof = _EXECRESPONSE.oneofs_by_name['args']
_HOMOGCALCREQUEST.fields_by_name['source_pts'].message_type = _FLOAT2DARRAY
_HOMOGCALCREQUEST.fields_by_name['dest_pts'].message_type = _FLOAT2DARRAY
_HOMOGCALCRESPONSE.fields_by_name['homography'].message_type = _FLOAT2DARRAY
_HOMOGWARPREQUEST.fields_by_name['image'].message_type = _IMAGE
_HOMOGWARPREQUEST.fields_by_name['homography'].message_type = _FLOAT2DARRAY
_HOMOGWARPRESPONSE.fields_by_name['image'].message_type = _IMAGE
DESCRIPTOR.message_types_by_name['Float1DArray'] = _FLOAT1DARRAY
DESCRIPTOR.message_types_by_name['Float2DArray'] = _FLOAT2DARRAY
DESCRIPTOR.message_types_by_name['Float3DArray'] = _FLOAT3DARRAY
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
DESCRIPTOR.message_types_by_name['ExecRequest'] = _EXECREQUEST
DESCRIPTOR.message_types_by_name['ExecResponse'] = _EXECRESPONSE
DESCRIPTOR.message_types_by_name['KltRequest'] = _KLTREQUEST
DESCRIPTOR.message_types_by_name['KltResponse'] = _KLTRESPONSE
DESCRIPTOR.message_types_by_name['HomogCalcRequest'] = _HOMOGCALCREQUEST
DESCRIPTOR.message_types_by_name['HomogCalcResponse'] = _HOMOGCALCRESPONSE
DESCRIPTOR.message_types_by_name['HomogWarpRequest'] = _HOMOGWARPREQUEST
DESCRIPTOR.message_types_by_name['HomogWarpResponse'] = _HOMOGWARPRESPONSE
DESCRIPTOR.message_types_by_name['SiftDetRequest'] = _SIFTDETREQUEST
DESCRIPTOR.message_types_by_name['SiftDetResponse'] = _SIFTDETRESPONSE
DESCRIPTOR.message_types_by_name['SiftMatchRequest'] = _SIFTMATCHREQUEST
DESCRIPTOR.message_types_by_name['SiftMatchResponse'] = _SIFTMATCHRESPONSE
DESCRIPTOR.message_types_by_name['LineDetRequest'] = _LINEDETREQUEST
DESCRIPTOR.message_types_by_name['LineDetResponse'] = _LINEDETRESPONSE
DESCRIPTOR.message_types_by_name['VidToFrameRequest'] = _VIDTOFRAMEREQUEST
DESCRIPTOR.message_types_by_name['VidToFrameResponse'] = _VIDTOFRAMERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Float1DArray = _reflection.GeneratedProtocolMessageType('Float1DArray', (_message.Message,), {
  'DESCRIPTOR' : _FLOAT1DARRAY,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:Float1DArray)
  })
_sym_db.RegisterMessage(Float1DArray)

Float2DArray = _reflection.GeneratedProtocolMessageType('Float2DArray', (_message.Message,), {
  'DESCRIPTOR' : _FLOAT2DARRAY,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:Float2DArray)
  })
_sym_db.RegisterMessage(Float2DArray)

Float3DArray = _reflection.GeneratedProtocolMessageType('Float3DArray', (_message.Message,), {
  'DESCRIPTOR' : _FLOAT3DARRAY,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:Float3DArray)
  })
_sym_db.RegisterMessage(Float3DArray)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), {
  'DESCRIPTOR' : _IMAGE,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:Image)
  })
_sym_db.RegisterMessage(Image)

ExecRequest = _reflection.GeneratedProtocolMessageType('ExecRequest', (_message.Message,), {
  'DESCRIPTOR' : _EXECREQUEST,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:ExecRequest)
  })
_sym_db.RegisterMessage(ExecRequest)

ExecResponse = _reflection.GeneratedProtocolMessageType('ExecResponse', (_message.Message,), {
  'DESCRIPTOR' : _EXECRESPONSE,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:ExecResponse)
  })
_sym_db.RegisterMessage(ExecResponse)

KltRequest = _reflection.GeneratedProtocolMessageType('KltRequest', (_message.Message,), {
  'DESCRIPTOR' : _KLTREQUEST,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:KltRequest)
  })
_sym_db.RegisterMessage(KltRequest)

KltResponse = _reflection.GeneratedProtocolMessageType('KltResponse', (_message.Message,), {
  'DESCRIPTOR' : _KLTRESPONSE,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:KltResponse)
  })
_sym_db.RegisterMessage(KltResponse)

HomogCalcRequest = _reflection.GeneratedProtocolMessageType('HomogCalcRequest', (_message.Message,), {
  'DESCRIPTOR' : _HOMOGCALCREQUEST,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:HomogCalcRequest)
  })
_sym_db.RegisterMessage(HomogCalcRequest)

HomogCalcResponse = _reflection.GeneratedProtocolMessageType('HomogCalcResponse', (_message.Message,), {
  'DESCRIPTOR' : _HOMOGCALCRESPONSE,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:HomogCalcResponse)
  })
_sym_db.RegisterMessage(HomogCalcResponse)

HomogWarpRequest = _reflection.GeneratedProtocolMessageType('HomogWarpRequest', (_message.Message,), {
  'DESCRIPTOR' : _HOMOGWARPREQUEST,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:HomogWarpRequest)
  })
_sym_db.RegisterMessage(HomogWarpRequest)

HomogWarpResponse = _reflection.GeneratedProtocolMessageType('HomogWarpResponse', (_message.Message,), {
  'DESCRIPTOR' : _HOMOGWARPRESPONSE,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:HomogWarpResponse)
  })
_sym_db.RegisterMessage(HomogWarpResponse)

SiftDetRequest = _reflection.GeneratedProtocolMessageType('SiftDetRequest', (_message.Message,), {
  'DESCRIPTOR' : _SIFTDETREQUEST,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:SiftDetRequest)
  })
_sym_db.RegisterMessage(SiftDetRequest)

SiftDetResponse = _reflection.GeneratedProtocolMessageType('SiftDetResponse', (_message.Message,), {
  'DESCRIPTOR' : _SIFTDETRESPONSE,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:SiftDetResponse)
  })
_sym_db.RegisterMessage(SiftDetResponse)

SiftMatchRequest = _reflection.GeneratedProtocolMessageType('SiftMatchRequest', (_message.Message,), {
  'DESCRIPTOR' : _SIFTMATCHREQUEST,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:SiftMatchRequest)
  })
_sym_db.RegisterMessage(SiftMatchRequest)

SiftMatchResponse = _reflection.GeneratedProtocolMessageType('SiftMatchResponse', (_message.Message,), {
  'DESCRIPTOR' : _SIFTMATCHRESPONSE,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:SiftMatchResponse)
  })
_sym_db.RegisterMessage(SiftMatchResponse)

LineDetRequest = _reflection.GeneratedProtocolMessageType('LineDetRequest', (_message.Message,), {
  'DESCRIPTOR' : _LINEDETREQUEST,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:LineDetRequest)
  })
_sym_db.RegisterMessage(LineDetRequest)

LineDetResponse = _reflection.GeneratedProtocolMessageType('LineDetResponse', (_message.Message,), {
  'DESCRIPTOR' : _LINEDETRESPONSE,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:LineDetResponse)
  })
_sym_db.RegisterMessage(LineDetResponse)

VidToFrameRequest = _reflection.GeneratedProtocolMessageType('VidToFrameRequest', (_message.Message,), {
  'DESCRIPTOR' : _VIDTOFRAMEREQUEST,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:VidToFrameRequest)
  })
_sym_db.RegisterMessage(VidToFrameRequest)

VidToFrameResponse = _reflection.GeneratedProtocolMessageType('VidToFrameResponse', (_message.Message,), {
  'DESCRIPTOR' : _VIDTOFRAMERESPONSE,
  '__module__' : 'vision_algorithms_pb2'
  # @@protoc_insertion_point(class_scope:VidToFrameResponse)
  })
_sym_db.RegisterMessage(VidToFrameResponse)



_VISIONALGORITHMS = _descriptor.ServiceDescriptor(
  name='VisionAlgorithms',
  full_name='VisionAlgorithms',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1452,
  serialized_end=1510,
  methods=[
  _descriptor.MethodDescriptor(
    name='Process',
    full_name='VisionAlgorithms.Process',
    index=0,
    containing_service=None,
    input_type=_EXECREQUEST,
    output_type=_EXECRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_VISIONALGORITHMS)

DESCRIPTOR.services_by_name['VisionAlgorithms'] = _VISIONALGORITHMS

# @@protoc_insertion_point(module_scope)
