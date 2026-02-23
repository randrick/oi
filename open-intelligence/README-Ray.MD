The original source code came from (https://github.com/norkator/open-intelligence).
                       
Norkator, the author, released the code under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 ("GA GPL").
The GA GPL, requires that all code changes be made publicly available under the same terms whenever
those changes are incorporated into publicly available or resides on a server that is accessible publicly.

The GA GPL, however does not require me to make my changes publicly available because:
>>"AGPL defines a user as anyone who accesses the server-side application if it is public-facing. For >>applications that reside inside the organization's network, AGPL does not trigger the release of the source >>code. (https://snyk.io/learn/agpl-license/)"

I have restructured, rewritten Dockerfiles, and replaced nearly all of the 'intelligence.'

More over I have done so without making any of making any of these changes available.  This is,
however, in compliance with this license.


Main table is called 'data'
    id                      detection_after_sr_completed
    name                    email_sent
    label                   insight_face_computed
    file_path               similarity_checked
    file_name               labeled_for_training
    file_create_date        labeling_image_x
    file_name_cropped       labeling_image_y
    detection_completed     labeling_image_x2
    detection_result        labeling_image_y2
    color                   deleted
    voice_completed         createdAt
    sr_image_computed       updatedAt 
    sr_image_name           

Todo:  move fields used for workflow state (ends in _completed or _computed)
       into seperate table









  