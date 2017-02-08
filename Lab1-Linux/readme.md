

# Introduction to Linux

##Question Answers
1.  What is the ``grep``command?

    The ``grep`` command allows to capture and filter incoming text into stdin by using a regular expression.

2.  What does the ``-prune`` option of ``find`` do? Give an example

    This option allows to restrict the search space to be explored. _e.g.,_ Restrict the search to files that match a regexp. For instance, the following example allows to search only PNG files starting on the file system root:
    ```bash
    find . -name / -prune -o -name '*.png' -print
    ```

3.  What does the ``cut`` command do?

    The ``cut`` command allows to select parts of lines, according to a set of delimiters, byte range, characters from a file and outputs it to stdout, this function also allows to take the complement of the search.

4.  What does the ``rsync`` command do?

    This command allows to copy files between two local/remote endpoints, this tool can update and copy the most recent modification of files, _e.g.,_ Suppose some folder A has a copy of all the files contained on a folder B, (Those folders can have a remote or local location). If some file in B folder has new modifications and some process/user wants to update the backup of B folder on A folder, ``rsync`` only updates the file that has new modifications, instead of copying all the files inside B folder.

5.  What does the ``diff`` command do?

    It allows to compare two files per a line basis, this command comes to help to compare two versions of the same file.

6.  What does the ``tail`` command do?

    ``tail`` allows to a process/user to print the last lines of a file, for instance, it can be used to print the last entries of a program log file.

7. What does the ``tail -f`` command do?

    It opens a continous real-time stream into stdout of the last lines of a file, it allows to see new lines that appear at the end of a file as they are appended.

8.  What does the ``link`` command do?

    It commands the creation of a hard link of a file which points to the reference node in the file system, opposed to ``ln``, which creates a soft link that points to the file pointer itself.

9. What is the meaning of ``#! /bin/bash`` at the start of scripts?

    The shebang line ``#! /bin/bash`` instructs to UNIX-like Operating Systems that the following file shall be executed using ``/bin/bash``. _i.e.,_ The file is a bash script. With this definition, a script which has execution permissions can be invoked without making an explicit call to ``sh <file>``. It is possible to define other shebang directives to be used with other interpreters or programs, for example, to execute Python scripts we can define ``#! /usr/bin/env python``

10.  How many users exist in the course server?

    All UNIX users are listed inside the ``/etc/passwd`` file, then, it is possible to count the number of users using ``wc`` as it follows:
    ```bash
    wc -l /etc/passwd
    ```

11. What command will produce a table of Users and Shells sorted by shell (tip: using ``cut`` and ``sort``)

    ```bash
    who | awk '{print $2, $1}' | sort
    ```

12. Create a script for finding duplicate images based on their content (tip: hash or checksum)

    See ``duplicates.sh <dir>``

14. What is the disk size of the uncompressed dataset, How many images are in the directory 'BSR/BSDS500/data/images'?

    After executing ``du -hs .``, the total size of the uncompressed folder corresponds to 73Mb. Finally, by executing `ls -lR ./BSR/BSDS500/data/images | grep jpg | wc -l``, it's possible to retrieve the total number of images present in the dataset, which corresponds to 500.

15. What is their resolution, what is their format?

    See ``image_info.sh <path_to_BSR_folder>``, this script is based on the ``identify`` function, included as part of ImageMagick.

16. How many of them are in *landscape* orientation (opposed to *portrait*)?

    After executing the script ``image_info.sh <path_to_BSR_folder>``, it is possible to conclude that the total number of portrait images is equal to 152, that implies that number of landscape images correspond to 348, when those two numbers are added together, the total result is equal to 500, the total number of images present in the dataset.

17. Crop all images to make them square (256x256).

    The script ``crop_images.sh -w <width> -h <height> <path_to_BSR_folder> <path_to_cropped_folder_storage>`` is based upon the tool ``convert``, included as part of the ImageMagick toolchain.

##References
[1] StackExchange Q&A Communities: SuperUser, StackOverflow and Ask Ubuntu

[2] Linux Man Pages, available at https://linux.die.net/man/

[3] ImageMagick Documentation, available at https://www.imagemagick.org/script/command-line-processing.php
