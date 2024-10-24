{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOkl1x6dgQvI9AZlDS5s3QS",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Gauri-Dublish/my-repository/blob/main/python_section_2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "def calculate_distance_matrix(df):\n",
        "    # Create a pivot table to get the distances between toll locations\n",
        "    distance_matrix = df.pivot_table(index='id_start', columns='id_end', values='distance', aggfunc='sum', fill_value=0)\n",
        "\n",
        "    # Ensure the matrix is symmetric\n",
        "    distance_matrix = distance_matrix + distance_matrix.T - np.diag(distance_matrix.values.diagonal())\n",
        "\n",
        "    # Set diagonal values to 0\n",
        "    np.fill_diagonal(distance_matrix.values, 0)\n",
        "\n",
        "    return distance_matrix\n",
        "\n",
        "# Example usage\n",
        "df = pd.read_csv('/content/dataset-2.csv')\n",
        "result = calculate_distance_matrix(df)\n",
        "print(result)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 353
        },
        "id": "krD9NbR-zluA",
        "outputId": "b62486de-5670-4113-f6d5-a1db1888e7a3"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ValueError",
          "evalue": "Unable to coerce to DataFrame, shape must be (43, 43): given (41, 41)",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-17-86cdf3a872e3>\u001b[0m in \u001b[0;36m<cell line: 18>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     16\u001b[0m \u001b[0;31m# Example usage\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     17\u001b[0m \u001b[0mdf\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'/content/dataset-2.csv'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 18\u001b[0;31m \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcalculate_distance_matrix\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdf\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     19\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m<ipython-input-17-86cdf3a872e3>\u001b[0m in \u001b[0;36mcalculate_distance_matrix\u001b[0;34m(df)\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      8\u001b[0m     \u001b[0;31m# Ensure the matrix is symmetric\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 9\u001b[0;31m     \u001b[0mdistance_matrix\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdistance_matrix\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0mdistance_matrix\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mT\u001b[0m \u001b[0;34m-\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdiag\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdistance_matrix\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdiagonal\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     10\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m     \u001b[0;31m# Set diagonal values to 0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/core/ops/common.py\u001b[0m in \u001b[0;36mnew_method\u001b[0;34m(self, other)\u001b[0m\n\u001b[1;32m     74\u001b[0m         \u001b[0mother\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mitem_from_zerodim\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mother\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     75\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 76\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mmethod\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mother\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     77\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     78\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mnew_method\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/core/arraylike.py\u001b[0m in \u001b[0;36m__sub__\u001b[0;34m(self, other)\u001b[0m\n\u001b[1;32m    192\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0munpack_zerodim_and_defer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"__sub__\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    193\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__sub__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mother\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 194\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_arith_method\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mother\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0moperator\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msub\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    195\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    196\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0munpack_zerodim_and_defer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"__rsub__\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36m_arith_method\u001b[0;34m(self, other, op)\u001b[0m\n\u001b[1;32m   7908\u001b[0m         \u001b[0mother\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mops\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmaybe_prepare_scalar_for_op\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mother\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0maxis\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   7909\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 7910\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mother\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_align_for_op\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mother\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mflex\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlevel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   7911\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   7912\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0merrstate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mall\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"ignore\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36m_align_for_op\u001b[0;34m(self, other, axis, flex, level)\u001b[0m\n\u001b[1;32m   8167\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   8168\u001b[0m                 \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 8169\u001b[0;31m                     raise ValueError(\n\u001b[0m\u001b[1;32m   8170\u001b[0m                         \u001b[0;34m\"Unable to coerce to DataFrame, shape \"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   8171\u001b[0m                         \u001b[0;34mf\"must be {left.shape}: given {right.shape}\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mValueError\u001b[0m: Unable to coerce to DataFrame, shape must be (43, 43): given (41, 41)"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "def unroll_distance_matrix(distance_matrix):\n",
        "    # Reset the index to get id_start and id_end as columns\n",
        "    unrolled_df = distance_matrix.reset_index().melt(id_vars='index', var_name='id_end', value_name='distance')\n",
        "\n",
        "    # Rename the columns\n",
        "    unrolled_df.rename(columns={'index': 'id_start'}, inplace=True)\n",
        "\n",
        "    # Remove rows where id_start is the same as id_end\n",
        "    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]\n",
        "\n",
        "    return unrolled_df\n",
        "\n",
        "# Example usage\n",
        "# Assuming `distance_matrix` is the DataFrame created in Question 9\n",
        "distance_matrix = pd.DataFrame({\n",
        "    'A': [0, 10, 20],\n",
        "    'B': [10, 0, 30],\n",
        "    'C': [20, 30, 0]\n",
        "}, index=['A', 'B', 'C'])\n",
        "\n",
        "result = unroll_distance_matrix(distance_matrix)\n",
        "print(result)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "uPWcB7RQyUvK",
        "outputId": "e5872931-3d66-48d5-db8f-279d993cd718"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  id_start id_end  distance\n",
            "1        B      A        10\n",
            "2        C      A        20\n",
            "3        A      B        10\n",
            "5        C      B        30\n",
            "6        A      C        20\n",
            "7        B      C        30\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "def find_ids_within_ten_percentage_threshold(df, reference_id):\n",
        "    # Calculate the average distance for the reference ID\n",
        "    reference_avg = df[df['id_start'] == reference_id]['distance'].mean()\n",
        "\n",
        "    # Calculate the 10% threshold\n",
        "    lower_bound = reference_avg * 0.9\n",
        "    upper_bound = reference_avg * 1.1\n",
        "\n",
        "    # Find IDs within the 10% threshold\n",
        "    ids_within_threshold = df.groupby('id_start')['distance'].mean()\n",
        "    ids_within_threshold = ids_within_threshold[(ids_within_threshold >= lower_bound) & (ids_within_threshold <= upper_bound)]\n",
        "\n",
        "    # Return the sorted list of IDs\n",
        "    return sorted(ids_within_threshold.index)\n",
        "\n",
        "# Example usage\n",
        "# Assuming `unrolled_df` is the DataFrame created in Question 10\n",
        "unrolled_df = pd.DataFrame({\n",
        "    'id_start': ['A', 'A', 'B', 'B', 'C', 'C'],\n",
        "    'id_end': ['B', 'C', 'A', 'C', 'A', 'B'],\n",
        "    'distance': [10, 20, 10, 30, 20, 30]\n",
        "})\n",
        "\n",
        "reference_id = 'A'\n",
        "result = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)\n",
        "print(result)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "aZYbWNrvyime",
        "outputId": "bb0dc988-3ee3-4780-8f8a-2de4957f9594"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['A']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "def calculate_toll_rate(df):\n",
        "    # Define rate coefficients for each vehicle type\n",
        "    rate_coefficients = {\n",
        "        'moto': 0.8,\n",
        "        'car': 1.2,\n",
        "        'rv': 1.5,\n",
        "        'bus': 2.2,\n",
        "        'truck': 3.6\n",
        "    }\n",
        "\n",
        "    # Calculate toll rates for each vehicle type and add as new columns\n",
        "    for vehicle, rate in rate_coefficients.items():\n",
        "        df[vehicle] = df['distance'] * rate\n",
        "\n",
        "    return df\n",
        "\n",
        "# Example usage\n",
        "# Assuming `unrolled_df` is the DataFrame created in Question 10\n",
        "unrolled_df = pd.DataFrame({\n",
        "    'id_start': ['A', 'A', 'B', 'B', 'C', 'C'],\n",
        "    'id_end': ['B', 'C', 'A', 'C', 'A', 'B'],\n",
        "    'distance': [10, 20, 10, 30, 20, 30]\n",
        "})\n",
        "\n",
        "result = calculate_toll_rate(unrolled_df)\n",
        "print(result)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "AqUc3C6KyvAX",
        "outputId": "128c7c9e-3a27-4c27-f343-44c7746e6338"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  id_start id_end  distance  moto   car    rv   bus  truck\n",
            "0        A      B        10   8.0  12.0  15.0  22.0   36.0\n",
            "1        A      C        20  16.0  24.0  30.0  44.0   72.0\n",
            "2        B      A        10   8.0  12.0  15.0  22.0   36.0\n",
            "3        B      C        30  24.0  36.0  45.0  66.0  108.0\n",
            "4        C      A        20  16.0  24.0  30.0  44.0   72.0\n",
            "5        C      B        30  24.0  36.0  45.0  66.0  108.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from datetime import datetime, time\n",
        "\n",
        "def calculate_time_based_toll_rates(df):\n",
        "    # Define discount factors for different time intervals\n",
        "    weekday_discounts = [\n",
        "        (time(0, 0), time(10, 0), 0.8),\n",
        "        (time(10, 0), time(18, 0), 1.2),\n",
        "        (time(18, 0), time(23, 59, 59), 0.8)\n",
        "    ]\n",
        "    weekend_discount = 0.7\n",
        "\n",
        "    # Function to apply discount based on time and day\n",
        "    def apply_discount(row):\n",
        "        start_time = row['start_time']\n",
        "        end_time = row['end_time']\n",
        "        start_day = row['start_day']\n",
        "        end_day = row['end_day']\n",
        "\n",
        "        # Determine if the day is a weekday or weekend\n",
        "        is_weekend = start_day in ['Saturday', 'Sunday']\n",
        "\n",
        "        # Apply weekend discount\n",
        "        if is_weekend:\n",
        "            for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:\n",
        "                row[vehicle] *= weekend_discount\n",
        "        else:\n",
        "            # Apply weekday discounts\n",
        "            for start, end, discount in weekday_discounts:\n",
        "                if start <= start_time < end:\n",
        "                    for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:\n",
        "                        row[vehicle] *= discount\n",
        "                    break\n",
        "        return row\n",
        "\n",
        "    # Add the new columns to the DataFrame\n",
        "    df['start_day'] = df['start'].dt.day_name()\n",
        "    df['start_time'] = df['start'].dt.time\n",
        "    df['end_day'] = df['end'].dt.day_name()\n",
        "    df['end_time'] = df['end'].dt.time\n",
        "\n",
        "    # Apply the discount function to each row\n",
        "    df = df.apply(apply_discount, axis=1)\n",
        "\n",
        "    return df\n",
        "\n",
        "# Example usage\n",
        "# Assuming `df` is the DataFrame created in Question 12\n",
        "df = pd.DataFrame({\n",
        "    'id_start': ['A', 'A', 'B', 'B', 'C', 'C'],\n",
        "    'id_end': ['B', 'C', 'A', 'C', 'A', 'B'],\n",
        "    'distance': [10, 20, 10, 30, 20, 30],\n",
        "    'moto': [8, 16, 8, 24, 16, 24],\n",
        "    'car': [12, 24, 12, 36, 24, 36],\n",
        "    'rv': [15, 30, 15, 45, 30, 45],\n",
        "    'bus': [22, 44, 22, 66, 44, 66],\n",
        "    'truck': [36, 72, 36, 108, 72, 108],\n",
        "    'start': pd.to_datetime(['2023-01-02 09:00:00', '2023-01-02 11:00:00', '2023-01-03 09:00:00', '2023-01-03 19:00:00', '2023-01-04 09:00:00', '2023-01-04 11:00:00']),\n",
        "    'end': pd.to_datetime(['2023-01-02 10:00:00', '2023-01-02 12:00:00', '2023-01-03 10:00:00', '2023-01-03 20:00:00', '2023-01-04 10:00:00', '2023-01-04 12:00:00'])\n",
        "})\n",
        "\n",
        "result = calculate_time_based_toll_rates(df)\n",
        "print(result)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "rMoQQX7UzQ8d",
        "outputId": "3b209098-57dc-43e3-bcb2-3ae293be3793"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  id_start id_end  distance  moto   car    rv   bus  truck  \\\n",
            "0        A      B        10   6.4   9.6  12.0  17.6   28.8   \n",
            "1        A      C        20  19.2  28.8  36.0  52.8   86.4   \n",
            "2        B      A        10   6.4   9.6  12.0  17.6   28.8   \n",
            "3        B      C        30  19.2  28.8  36.0  52.8   86.4   \n",
            "4        C      A        20  12.8  19.2  24.0  35.2   57.6   \n",
            "5        C      B        30  28.8  43.2  54.0  79.2  129.6   \n",
            "\n",
            "                start                 end  start_day start_time    end_day  \\\n",
            "0 2023-01-02 09:00:00 2023-01-02 10:00:00     Monday   09:00:00     Monday   \n",
            "1 2023-01-02 11:00:00 2023-01-02 12:00:00     Monday   11:00:00     Monday   \n",
            "2 2023-01-03 09:00:00 2023-01-03 10:00:00    Tuesday   09:00:00    Tuesday   \n",
            "3 2023-01-03 19:00:00 2023-01-03 20:00:00    Tuesday   19:00:00    Tuesday   \n",
            "4 2023-01-04 09:00:00 2023-01-04 10:00:00  Wednesday   09:00:00  Wednesday   \n",
            "5 2023-01-04 11:00:00 2023-01-04 12:00:00  Wednesday   11:00:00  Wednesday   \n",
            "\n",
            "   end_time  \n",
            "0  10:00:00  \n",
            "1  12:00:00  \n",
            "2  10:00:00  \n",
            "3  20:00:00  \n",
            "4  10:00:00  \n",
            "5  12:00:00  \n"
          ]
        }
      ]
    }
  ]
}